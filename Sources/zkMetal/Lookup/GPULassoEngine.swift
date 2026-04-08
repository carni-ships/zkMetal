// GPULassoEngine — GPU-accelerated Lasso sparse lookup argument
//
// Implements the Lasso lookup argument (Setty et al. 2023) with GPU acceleration
// for the core memory-checking operations.
//
// Key idea: Lasso decomposes large lookup tables T into small subtables
// T = T1 x T2 x ... x Tc via tensor structure. For each subtable, offline
// memory checking verifies that every accessed index was valid:
//
//   For subtable Tk of size Sk with m lookups:
//     read_ts[i]  = read timestamp for lookup i
//     write_ts[j] = write timestamp for table entry j
//     final_ts[j] = final timestamp for table entry j
//
//   Memory checking invariant:
//     sum_i 1/(gamma + read_ts[i]) + sum_j 1/(gamma + write_ts[j])
//       = sum_j 1/(gamma + final_ts[j]) + sum_j 1/(gamma + init_ts[j])
//
// This reduces to the simpler multiplicity check:
//     sum_i 1/(beta + v_k[i]) = sum_j count[j]/(beta + Tk[j])
//
// GPU acceleration targets:
//   - Batch inverse via GPUBatchInverseEngine (Montgomery's trick)
//   - Multilinear extension evaluation for read/write timestamps
//   - Subtable decomposition (parallel)
//   - Memory check sum accumulation
//
// References:
//   Lasso: Setty et al. 2023 (eprint 2023/1216)
//   Jolt: Arun et al. 2024

import Foundation
import Metal
import NeonFieldOps

// MARK: - Structured Table Types

/// Structured table type for GPU Lasso engine.
public enum GPULassoTableType {
    /// Range table: values in [0, 2^bits), decomposed into chunks of bitsPerChunk bits
    case range(bits: Int, chunks: Int)
    /// Bitwise AND: z = x AND y for given bit-width
    case bitwiseAnd(bits: Int)
    /// Bitwise XOR: z = x XOR y for given bit-width
    case bitwiseXor(bits: Int)
    /// Custom table with explicit subtables and decomposition
    case custom(subtables: [[Fr]], decompose: (Fr) -> [Int], compose: ([Fr]) -> Fr)
}

// MARK: - GPU Lasso Table Definition

/// Defines a structured lookup table for GPU Lasso, with subtable decomposition.
public struct GPULassoTable {
    /// Subtables: subtables[k] is the k-th subtable
    public let subtables: [[Fr]]
    /// Number of chunks (subtables)
    public let numChunks: Int
    /// Decompose a value into per-subtable indices
    public let decompose: (Fr) -> [Int]
    /// Compose subtable values back into a full value
    public let compose: ([Fr]) -> Fr

    /// Build a GPU Lasso table from a type descriptor.
    public static func build(_ type: GPULassoTableType) -> GPULassoTable {
        switch type {
        case .range(let bits, let chunks):
            return buildRange(bits: bits, chunks: chunks)
        case .bitwiseAnd(let bits):
            return buildBitwiseOp(bits: bits, op: { $0 & $1 })
        case .bitwiseXor(let bits):
            return buildBitwiseOp(bits: bits, op: { $0 ^ $1 })
        case .custom(let subtables, let decompose, let compose):
            return GPULassoTable(subtables: subtables, numChunks: subtables.count,
                                 decompose: decompose, compose: compose)
        }
    }

    private static func buildRange(bits: Int, chunks: Int) -> GPULassoTable {
        precondition(bits > 0 && bits % chunks == 0, "bits must be divisible by chunks")
        let bitsPerChunk = bits / chunks
        let chunkSize = 1 << bitsPerChunk
        let chunkMask = UInt64(chunkSize - 1)

        let subtable: [Fr] = (0..<chunkSize).map { frFromInt(UInt64($0)) }
        let subtables = Array(repeating: subtable, count: chunks)

        let compose: ([Fr]) -> Fr = { components in
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
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(chunks)
            for k in 0..<chunks {
                let shift = k * bitsPerChunk
                indices.append(Int((v >> shift) & chunkMask))
            }
            return indices
        }

        return GPULassoTable(subtables: subtables, numChunks: chunks,
                             decompose: decompose, compose: compose)
    }

    private static func buildBitwiseOp(bits: Int, op: @escaping (UInt64, UInt64) -> UInt64) -> GPULassoTable {
        precondition(bits > 0 && bits % 2 == 0, "bits must be even")
        let bitsPerChunk = min(bits, 8)
        let chunks = bits / bitsPerChunk
        let chunkRange = 1 << bitsPerChunk
        let chunkMask = UInt64(chunkRange - 1)

        let tableSize = chunkRange * chunkRange
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = UInt64(idx / chunkRange)
            let b = UInt64(idx % chunkRange)
            return frFromInt(op(a, b))
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
                indices.append(Int((v >> shift) & chunkMask))
            }
            return indices
        }

        return GPULassoTable(subtables: subtables, numChunks: chunks,
                             decompose: decompose, compose: compose)
    }
}

// MARK: - Proof Structures

/// Per-subtable memory checking proof.
public struct GPUSubtableMemoryProof {
    /// Chunk index
    public let chunkIndex: Int
    /// Read counts: readCounts[j] = number of times subtable[j] was accessed
    public let readCounts: [Fr]
    /// Challenge beta
    public let beta: Fr
    /// Read-side inverses: 1/(beta + v_k[i]) for each lookup i
    public let readInverses: [Fr]
    /// Table-side terms: readCounts[j]/(beta + Tk[j])
    public let tableTerms: [Fr]
    /// Read-side sum: sum_i 1/(beta + v_k[i])
    public let readSum: Fr
    /// Table-side sum: sum_j readCounts[j]/(beta + Tk[j])
    public let tableSum: Fr

    public init(chunkIndex: Int, readCounts: [Fr], beta: Fr,
                readInverses: [Fr], tableTerms: [Fr],
                readSum: Fr, tableSum: Fr) {
        self.chunkIndex = chunkIndex
        self.readCounts = readCounts
        self.beta = beta
        self.readInverses = readInverses
        self.tableTerms = tableTerms
        self.readSum = readSum
        self.tableSum = tableSum
    }
}

/// Complete GPU Lasso proof.
public struct GPULassoProof {
    /// Number of chunks (subtables)
    public let numChunks: Int
    /// Per-subtable memory proofs
    public let subtableProofs: [GPUSubtableMemoryProof]
    /// Decomposed indices: indices[k][i] = index into subtable k for lookup i
    public let indices: [[Int]]
    /// Whether all memory checks passed (read sum == table sum for all chunks)
    public let memoryCheckPassed: Bool

    public init(numChunks: Int, subtableProofs: [GPUSubtableMemoryProof],
                indices: [[Int]], memoryCheckPassed: Bool) {
        self.numChunks = numChunks
        self.subtableProofs = subtableProofs
        self.indices = indices
        self.memoryCheckPassed = memoryCheckPassed
    }
}

// MARK: - GPU Lasso Engine

/// GPU-accelerated Lasso sparse lookup engine.
///
/// Uses GPUBatchInverseEngine for the core memory-checking inverse computations,
/// and supports structured table decomposition (range, bitwise ops).
public class GPULassoEngine {
    public static let version = Versions.gpuLasso

    private let inverseEngine: GPUBatchInverseEngine

    /// Emit profiling info to stderr
    public var profile = false

    /// GPU threshold: arrays smaller than this use CPU batch inverse path
    public var gpuThreshold: Int = 256

    public init() throws {
        self.inverseEngine = try GPUBatchInverseEngine()
    }

    // MARK: - Prove

    /// Prove that every element in `lookups` exists in the structured table.
    ///
    /// - Parameters:
    ///   - lookups: The values to look up. Each must be in the table.
    ///   - table: A GPULassoTable defining the structured decomposition.
    ///   - beta: Random challenge. Pass nil for Fiat-Shamir derivation.
    /// - Returns: A GPULassoProof containing per-subtable memory proofs.
    public func prove(lookups: [Fr], table: GPULassoTable,
                      beta: Fr? = nil) throws -> GPULassoProof {
        let m = lookups.count
        precondition(m > 0, "Lookups must be non-empty")

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Decompose each lookup into subtable indices
        var indices = [[Int]](repeating: [Int](repeating: 0, count: m), count: table.numChunks)
        for i in 0..<m {
            let decomposed = table.decompose(lookups[i])
            precondition(decomposed.count == table.numChunks,
                         "Decomposition must produce \(table.numChunks) indices, got \(decomposed.count)")
            for k in 0..<table.numChunks {
                indices[k][i] = decomposed[k]
            }
        }

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-lasso] decompose (%d lookups, %d chunks): %.2f ms\n",
                         m, table.numChunks, (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 2: Derive challenge
        let betaVal = beta ?? deriveBeta(lookups: lookups, table: table)

        // Step 3: For each subtable, run memory checking
        var subtableProofs = [GPUSubtableMemoryProof]()
        subtableProofs.reserveCapacity(table.numChunks)
        var allPassed = true

        for k in 0..<table.numChunks {
            if profile { _tPhase = CFAbsoluteTimeGetCurrent() }

            let subtable = table.subtables[k]
            let S = subtable.count

            // 3a: Compute read counts
            var readCounts = [Fr](repeating: Fr.zero, count: S)
            for i in 0..<m {
                let idx = indices[k][i]
                precondition(idx >= 0 && idx < S,
                             "Index \(idx) out of range for subtable \(k) of size \(S)")
                readCounts[idx] = frAdd(readCounts[idx], Fr.one)
            }

            if profile {
                let _t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [gpu-lasso] chunk %d read_counts: %.2f ms\n",
                             k, (_t - _tPhase) * 1000), stderr)
                _tPhase = _t
            }

            // 3b: Compute read-side denominators: beta + v_k[i]
            // where v_k[i] = subtable[indices[k][i]]
            var readDenoms = [Fr](repeating: Fr.zero, count: m)
            for i in 0..<m {
                readDenoms[i] = frAdd(betaVal, subtable[indices[k][i]])
            }

            // 3c: Compute table-side denominators: beta + Tk[j]
            var tableDenoms = [Fr](repeating: Fr.zero, count: S)
            for j in 0..<S {
                tableDenoms[j] = frAdd(betaVal, subtable[j])
            }

            if profile {
                let _t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [gpu-lasso] chunk %d denoms: %.2f ms\n",
                             k, (_t - _tPhase) * 1000), stderr)
                _tPhase = _t
            }

            // 3d: GPU batch inverse
            let readInvs = try inverseEngine.batchInverseFr(readDenoms)
            let tableInvs = try inverseEngine.batchInverseFr(tableDenoms)

            if profile {
                let _t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [gpu-lasso] chunk %d batch_inverse (m=%d, S=%d): %.2f ms\n",
                             k, m, S, (_t - _tPhase) * 1000), stderr)
                _tPhase = _t
            }

            // 3e: Compute table terms: readCounts[j] * tableInvs[j]
            var tableTerms = [Fr](repeating: Fr.zero, count: S)
            for j in 0..<S {
                tableTerms[j] = frMul(readCounts[j], tableInvs[j])
            }

            // 3f: Compute sums
            var readSum = Fr.zero
            for i in 0..<m {
                readSum = frAdd(readSum, readInvs[i])
            }

            var tableSum = Fr.zero
            for j in 0..<S {
                tableSum = frAdd(tableSum, tableTerms[j])
            }

            let chunkPassed = frEqual(readSum, tableSum)
            if !chunkPassed { allPassed = false }

            if profile {
                let _t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [gpu-lasso] chunk %d sums (match=%@): %.2f ms\n",
                             k, chunkPassed ? "YES" : "NO", (_t - _tPhase) * 1000), stderr)
                _tPhase = _t
            }

            subtableProofs.append(GPUSubtableMemoryProof(
                chunkIndex: k,
                readCounts: readCounts,
                beta: betaVal,
                readInverses: readInvs,
                tableTerms: tableTerms,
                readSum: readSum,
                tableSum: tableSum
            ))
        }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-lasso] TOTAL prove: %.2f ms (passed=%@)\n",
                         total, allPassed ? "YES" : "NO"), stderr)
        }

        return GPULassoProof(
            numChunks: table.numChunks,
            subtableProofs: subtableProofs,
            indices: indices,
            memoryCheckPassed: allPassed
        )
    }

    // MARK: - Verify

    /// Verify a GPU Lasso proof.
    ///
    /// Checks for each subtable:
    ///   1. Read counts are consistent with decomposed indices.
    ///   2. Each read inverse h_r[i] = 1/(beta + v_k[i]).
    ///   3. Each table term h_t[j] = readCounts[j]/(beta + Tk[j]).
    ///   4. Read sum equals table sum (memory check).
    ///   5. Decomposed indices reconstruct the original lookup values.
    ///
    /// - Parameters:
    ///   - proof: The GPULassoProof to verify.
    ///   - lookups: The original lookup values.
    ///   - table: The GPULassoTable.
    /// - Returns: true if the proof is valid.
    public func verify(proof: GPULassoProof, lookups: [Fr],
                       table: GPULassoTable) -> Bool {
        let m = lookups.count
        guard proof.numChunks == table.numChunks else { return false }
        guard proof.subtableProofs.count == table.numChunks else { return false }
        guard proof.indices.count == table.numChunks else { return false }

        // Check: decomposed indices reconstruct the original lookup values
        for i in 0..<m {
            var components = [Fr]()
            components.reserveCapacity(table.numChunks)
            for k in 0..<table.numChunks {
                guard proof.indices[k].count == m else { return false }
                let idx = proof.indices[k][i]
                guard idx >= 0 && idx < table.subtables[k].count else { return false }
                components.append(table.subtables[k][idx])
            }
            let reconstructed = table.compose(components)
            if !frEqual(reconstructed, lookups[i]) { return false }
        }

        // Check each subtable proof
        for k in 0..<table.numChunks {
            let sp = proof.subtableProofs[k]
            let subtable = table.subtables[k]
            let S = subtable.count
            let beta = sp.beta

            // Memory check: read sum == table sum
            guard frEqual(sp.readSum, sp.tableSum) else { return false }

            // Read counts consistency
            guard sp.readCounts.count == S else { return false }
            var recomputedCounts = [Fr](repeating: Fr.zero, count: S)
            for i in 0..<m {
                let idx = proof.indices[k][i]
                recomputedCounts[idx] = frAdd(recomputedCounts[idx], Fr.one)
            }
            for j in 0..<S {
                if !frEqual(recomputedCounts[j], sp.readCounts[j]) { return false }
            }

            // Verify read inverses: h_r[i] * (beta + v_k[i]) == 1
            guard sp.readInverses.count == m else { return false }
            for i in 0..<m {
                let denom = frAdd(beta, subtable[proof.indices[k][i]])
                let product = frMul(sp.readInverses[i], denom)
                if !frEqual(product, Fr.one) { return false }
            }

            // Verify table terms: h_t[j] * (beta + Tk[j]) == readCounts[j]
            guard sp.tableTerms.count == S else { return false }
            for j in 0..<S {
                let denom = frAdd(beta, subtable[j])
                let product = frMul(sp.tableTerms[j], denom)
                if !frEqual(product, sp.readCounts[j]) { return false }
            }

            // Recompute sums
            var readSum = Fr.zero
            for i in 0..<m {
                readSum = frAdd(readSum, sp.readInverses[i])
            }
            guard frEqual(readSum, sp.readSum) else { return false }

            var tableSum = Fr.zero
            for j in 0..<S {
                tableSum = frAdd(tableSum, sp.tableTerms[j])
            }
            guard frEqual(tableSum, sp.tableSum) else { return false }
        }

        return true
    }

    // MARK: - Multilinear Extension Evaluation

    /// Evaluate the multilinear extension of a vector at a random point.
    /// MLE(r) = sum_i f[i] * prod_j (r_j * bit_j(i) + (1 - r_j) * (1 - bit_j(i)))
    ///
    /// Used for read/write timestamp polynomial evaluation in sumcheck integration.
    ///
    /// - Parameters:
    ///   - evals: The evaluation vector (length must be power of 2).
    ///   - point: The evaluation point (length = log2(evals.count)).
    /// - Returns: The MLE evaluation at the given point.
    public func evaluateMultilinearExtension(evals: [Fr], point: [Fr]) -> Fr {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Length must be power of 2")
        let numVars = point.count
        precondition(1 << numVars == n, "Point dimension must match log2(n)")

        // Iterative halving: fold evals along each variable
        var current = evals
        var size = n
        for v in 0..<numVars {
            let half = size / 2
            let r = point[v]
            current.withUnsafeMutableBytes { cBuf in
                withUnsafeBytes(of: r) { rBuf in
                    bn254_fr_fold_interleaved_inplace(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
            }
            current.removeLast(half)
            size = half
        }
        return current[0]
    }

    // MARK: - Timestamp-Based Memory Checking

    /// Perform offline memory checking with timestamps for a subtable.
    ///
    /// This is the full Lasso memory checking protocol:
    ///   - init_ts[j] = j (initial timestamp)
    ///   - For each access i: read_ts[i] = current_ts[addr[i]], then write_ts[i] = read_ts[i] + 1
    ///   - final_ts[j] = last written timestamp for address j
    ///
    /// Memory check: multiset equality of (addr, ts) tuples:
    ///   {(addr[i], read_ts[i])} union {(j, init_ts[j])} = {(addr[i], write_ts[i])} union {(j, final_ts[j])}
    ///
    /// Returns (readTimestamps, writeTimestamps, finalTimestamps).
    public func computeTimestamps(indices: [Int], subtableSize: Int)
        -> (readTs: [Fr], writeTs: [Fr], finalTs: [Fr]) {
        let m = indices.count

        var currentTs = [UInt64](repeating: 0, count: subtableSize)
        var readTs = [Fr](repeating: Fr.zero, count: m)
        var writeTs = [Fr](repeating: Fr.zero, count: m)

        for i in 0..<m {
            let addr = indices[i]
            readTs[i] = frFromInt(currentTs[addr])
            currentTs[addr] += 1
            writeTs[i] = frFromInt(currentTs[addr])
        }

        let finalTs: [Fr] = currentTs.map { frFromInt($0) }
        return (readTs, writeTs, finalTs)
    }

    /// Verify timestamp-based memory checking using random challenge.
    ///
    /// Checks the multiset equality:
    ///   sum_i 1/(gamma + encode(addr[i], read_ts[i])) + sum_j 1/(gamma + encode(j, init_ts[j]))
    ///     = sum_i 1/(gamma + encode(addr[i], write_ts[i])) + sum_j 1/(gamma + encode(j, final_ts[j]))
    ///
    /// where encode(addr, ts) = addr + alpha * ts for a random alpha.
    public func verifyTimestampMemoryCheck(
        indices: [Int], subtable: [Fr],
        readTs: [Fr], writeTs: [Fr], finalTs: [Fr],
        gamma: Fr, alpha: Fr
    ) throws -> Bool {
        let m = indices.count
        let S = subtable.count

        // Encode: val = subtable[addr] + alpha * ts
        func encode(_ addr: Int, _ ts: Fr) -> Fr {
            return frAdd(subtable[addr], frMul(alpha, ts))
        }

        // Read side: sum_i 1/(gamma + encode(addr[i], readTs[i]))
        var readDenoms = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            readDenoms[i] = frAdd(gamma, encode(indices[i], readTs[i]))
        }

        // Write side: sum_i 1/(gamma + encode(addr[i], writeTs[i]))
        var writeDenoms = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            writeDenoms[i] = frAdd(gamma, encode(indices[i], writeTs[i]))
        }

        // Init side: sum_j 1/(gamma + encode(j, 0))
        var initDenoms = [Fr](repeating: Fr.zero, count: S)
        for j in 0..<S {
            initDenoms[j] = frAdd(gamma, encode(j, Fr.zero))
        }

        // Final side: sum_j 1/(gamma + encode(j, finalTs[j]))
        var finalDenoms = [Fr](repeating: Fr.zero, count: S)
        for j in 0..<S {
            finalDenoms[j] = frAdd(gamma, encode(j, finalTs[j]))
        }

        // GPU batch inverse all four sets
        let readInvs = try inverseEngine.batchInverseFr(readDenoms)
        let writeInvs = try inverseEngine.batchInverseFr(writeDenoms)
        let initInvs = try inverseEngine.batchInverseFr(initDenoms)
        let finalInvs = try inverseEngine.batchInverseFr(finalDenoms)

        // LHS = sum(readInvs) + sum(initInvs)
        var lhs = Fr.zero
        for v in readInvs { lhs = frAdd(lhs, v) }
        for v in initInvs { lhs = frAdd(lhs, v) }

        // RHS = sum(writeInvs) + sum(finalInvs)
        var rhs = Fr.zero
        for v in writeInvs { rhs = frAdd(rhs, v) }
        for v in finalInvs { rhs = frAdd(rhs, v) }

        return frEqual(lhs, rhs)
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveBeta(lookups: [Fr], table: GPULassoTable) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x47, 0x4C, 0x53, 0x4F]) // "GLSO"
        appendSizeAndSamples(&transcript, lookups)
        for st in table.subtables {
            appendSizeAndSamples(&transcript, st)
        }
        return hashToFr(transcript)
    }

    private func appendSizeAndSamples(_ transcript: inout [UInt8], _ arr: [Fr]) {
        var size = UInt64(arr.count)
        for _ in 0..<8 {
            transcript.append(UInt8(size & 0xFF))
            size >>= 8
        }
        let sampleCount = min(8, arr.count)
        for i in 0..<sampleCount {
            appendFrBytes(&transcript, arr[i])
        }
    }

    private func appendFrBytes(_ transcript: inout [UInt8], _ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func hashToFr(_ data: [UInt8]) -> Fr {
        let hash = blake3(data)
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
