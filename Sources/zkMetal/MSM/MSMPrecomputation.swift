// MSM Precomputation — fixed-base window tables and BGMW-style lookup MSM
// For SRS/generator points that are reused across many MSM calls,
// precomputing window tables eliminates all doublings at commit time.
//
// References: Bos-Gordon-Mundy-Wahby (BGMW) fixed-base scalar multiplication
//             "Faster modular exponentiation using double precision floating point"

import Foundation
import NeonFieldOps

// MARK: - Precomputed Window Table

/// Precomputed window tables for fixed-base MSM.
/// For each base point g_i, stores [1*g_i, 2*g_i, ..., (2^w-1)*g_i] for each window position.
/// At MSM time, each scalar is decomposed into base-2^w digits and the result is a
/// pure-addition accumulation (no doublings).
public struct MSMPrecomputedTable {
    /// The precomputed affine points, stored flat:
    /// table[i * (numWindows * tableSize) + w * tableSize + (d-1)]
    /// = d * 2^(w * windowBits) * points[i]
    public let table: [PointAffine]

    /// Number of base points
    public let pointCount: Int

    /// Window width in bits
    public let windowBits: Int

    /// Number of windows: ceil(scalarBits / windowBits)
    public let numWindows: Int

    /// Entries per window: 2^windowBits - 1 (digits 1..2^w-1)
    public let tableSize: Int

    /// Total table entries: pointCount * numWindows * tableSize
    public var totalEntries: Int { pointCount * numWindows * tableSize }

    /// Approximate memory usage in bytes
    public var memorySizeBytes: Int {
        totalEntries * MemoryLayout<PointAffine>.stride
    }
}

// MARK: - Precomputation

/// Precompute BGMW window tables for a set of fixed base points.
/// Uses the C implementation for fast multi-threaded precomputation.
///
/// - Parameters:
///   - points: base points in affine form (e.g., SRS points)
///   - windowBits: window width (default 7, good for 256-bit scalars)
///   - scalarBits: number of scalar bits (default 256 for BN254 Fr)
/// - Returns: precomputed table ready for bgmwMSM calls
public func precomputeWindowTable(
    points: [PointAffine],
    windowBits: Int = 7,
    scalarBits: Int = 256
) -> MSMPrecomputedTable {
    let n = points.count
    let numWindows = (scalarBits + windowBits - 1) / windowBits
    let tableSize = (1 << windowBits) - 1
    let totalEntries = n * numWindows * tableSize

    var affineTable = [PointAffine](repeating: PointAffine(x: .one, y: .one), count: totalEntries)
    points.withUnsafeBytes { ptsBuf in
        affineTable.withUnsafeMutableBytes { tableBuf in
            bgmw_precompute(
                ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(n),
                Int32(windowBits),
                tableBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
        }
    }

    return MSMPrecomputedTable(
        table: affineTable,
        pointCount: n,
        windowBits: windowBits,
        numWindows: numWindows,
        tableSize: tableSize
    )
}

// MARK: - BGMW MSM using precomputed tables

/// Fixed-base MSM using precomputed BGMW window tables.
/// Decomposes each scalar into base-2^w digits and sums table lookups.
/// Runtime is pure additions (no doublings), making this significantly
/// faster than variable-base Pippenger for the same point set.
///
/// - Parameters:
///   - table: precomputed window table from precomputeWindowTable()
///   - scalars: scalar vectors as UInt32 limbs (8 limbs per scalar, little-endian integer form)
/// - Returns: MSM result as projective point
public func bgmwMSM(
    table: MSMPrecomputedTable,
    scalars: [[UInt32]]
) -> PointProjective {
    let n = scalars.count
    precondition(n == table.pointCount, "Scalar count must match precomputed table point count")
    if n == 0 { return pointIdentity() }

    // Flatten scalars
    var flatScalars = [UInt32]()
    flatScalars.reserveCapacity(n * 8)
    for s in scalars { flatScalars.append(contentsOf: s) }

    var result = PointProjective(x: .one, y: .one, z: .zero)
    table.table.withUnsafeBytes { tableBuf in
        flatScalars.withUnsafeBufferPointer { scalarBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bgmw_msm(
                    tableBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    Int32(table.windowBits),
                    scalarBuf.baseAddress!,
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }
    }
    return result
}

/// Fixed-base MSM with Fr scalars (handles Montgomery-to-limb conversion).
public func bgmwMSM(
    table: MSMPrecomputedTable,
    frScalars: [Fr]
) -> PointProjective {
    let n = frScalars.count
    precondition(n == table.pointCount)
    if n == 0 { return pointIdentity() }

    // Batch convert Montgomery Fr to integer limbs
    var flatScalars = [UInt32](repeating: 0, count: n * 8)
    frScalars.withUnsafeBytes { sBuf in
        flatScalars.withUnsafeMutableBufferPointer { lBuf in
            bn254_fr_batch_to_limbs(
                sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                lBuf.baseAddress!,
                Int32(n))
        }
    }

    var result = PointProjective(x: .one, y: .one, z: .zero)
    table.table.withUnsafeBytes { tableBuf in
        flatScalars.withUnsafeBufferPointer { scalarBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bgmw_msm(
                    tableBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    Int32(table.windowBits),
                    scalarBuf.baseAddress!,
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }
    }
    return result
}

/// Multi-commit using precomputed tables: compute k MSMs with the same fixed bases.
/// Each MSM uses the precomputed table (no doublings), giving significant speedup
/// over variable-base multiMSM for repeated SRS commitments.
public func bgmwMultiMSM(
    table: MSMPrecomputedTable,
    scalarSets: [[[UInt32]]]
) -> [PointProjective] {
    return scalarSets.map { scalars in
        bgmwMSM(table: table, scalars: scalars)
    }
}

// MARK: - Serialization

/// Serialize a precomputed table to a binary file for caching.
/// Format: [pointCount: UInt32][windowBits: UInt32][numWindows: UInt32][tableSize: UInt32]
///         [table data: totalEntries * PointAffine]
public func serializePrecomputedTable(_ table: MSMPrecomputedTable, to url: URL) throws {
    var data = Data()
    data.reserveCapacity(16 + table.totalEntries * MemoryLayout<PointAffine>.stride)

    // Header
    var pc = UInt32(table.pointCount)
    var wb = UInt32(table.windowBits)
    var nw = UInt32(table.numWindows)
    var ts = UInt32(table.tableSize)
    data.append(Data(bytes: &pc, count: 4))
    data.append(Data(bytes: &wb, count: 4))
    data.append(Data(bytes: &nw, count: 4))
    data.append(Data(bytes: &ts, count: 4))

    // Table data
    table.table.withUnsafeBytes { buf in
        data.append(Data(buf))
    }

    try data.write(to: url)
}

/// Deserialize a precomputed table from a binary file.
public func deserializePrecomputedTable(from url: URL) throws -> MSMPrecomputedTable {
    let data = try Data(contentsOf: url)
    guard data.count >= 16 else {
        throw MSMError.invalidInput
    }

    let pc = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
    let wb = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
    let nw = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) }
    let ts = data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt32.self) }

    let totalEntries = Int(pc) * Int(nw) * Int(ts)
    let expectedSize = 16 + totalEntries * MemoryLayout<PointAffine>.stride
    guard data.count == expectedSize else {
        throw MSMError.invalidInput
    }

    var table = [PointAffine](repeating: PointAffine(x: .one, y: .one), count: totalEntries)
    data.withUnsafeBytes { buf in
        let src = buf.baseAddress!.advanced(by: 16)
        table.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, src, totalEntries * MemoryLayout<PointAffine>.stride)
        }
    }

    return MSMPrecomputedTable(
        table: table,
        pointCount: Int(pc),
        windowBits: Int(wb),
        numWindows: Int(nw),
        tableSize: Int(ts)
    )
}

// MARK: - Cached precomputation with automatic persistence

/// Cache directory for precomputed tables
private let precomputeCacheDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".zkmsm").appendingPathComponent("precompute")

/// Get or compute a precomputed table, with automatic file caching.
/// The cache key is derived from the number of points and window bits.
/// If a cached table exists with matching parameters, it is loaded from disk.
/// Otherwise, the table is computed and saved for future use.
///
/// - Parameters:
///   - points: base points in affine form
///   - windowBits: window width (default 7)
///   - cacheKey: optional string to disambiguate different point sets of the same size
/// - Returns: precomputed table (from cache or freshly computed)
public func getOrPrecomputeTable(
    points: [PointAffine],
    windowBits: Int = 7,
    cacheKey: String? = nil
) -> MSMPrecomputedTable {
    let key = cacheKey ?? "bn254_n\(points.count)_w\(windowBits)"
    let cacheFile = precomputeCacheDir.appendingPathComponent("\(key).bgmw")

    // Try to load from cache
    if let cached = try? deserializePrecomputedTable(from: cacheFile),
       cached.pointCount == points.count,
       cached.windowBits == windowBits {
        return cached
    }

    // Compute fresh
    let table = precomputeWindowTable(points: points, windowBits: windowBits)

    // Save to cache (best effort)
    try? FileManager.default.createDirectory(
        at: precomputeCacheDir, withIntermediateDirectories: true)
    try? serializePrecomputedTable(table, to: cacheFile)

    return table
}
