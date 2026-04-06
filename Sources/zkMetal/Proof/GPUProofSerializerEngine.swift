// GPUProofSerializerEngine — GPU-accelerated proof serialization engine
//
// Efficient proof serialization/deserialization with:
//   - Compact binary format for field elements, group points, Merkle paths
//   - Pattern compression for repeated proof data (run-length + dedup)
//   - Streaming serialization (progressive buffer writes)
//   - Format versioning for forward/backward compatibility
//   - Proof size estimation before serialization
//   - Support for Groth16, Plonk, STARK, and FRI proof formats
//
// Wire format:
//   Header (20 bytes):
//     [0..3]   Magic: "ZKSZ" (0x5A 0x4B 0x53 0x5A)
//     [4..5]   Format version (u16 LE)
//     [6]      ProofSystemType (u8)
//     [7]      CurveId (u8)
//     [8]      Compression flags (u8)
//     [9..12]  Uncompressed size (u32 LE)
//     [13..16] Compressed size (u32 LE)
//     [17..19] Reserved (3 bytes, zero)
//
//   Body: sequence of tagged sections, each:
//     [0]      SectionTag (u8)
//     [1..4]   Section length (u32 LE)
//     [5..]    Section data
//
// Works with BN254 Fr/Fp field types and PointProjective/PointAffine.

import Foundation
import NeonFieldOps

// MARK: - Format Constants

/// Magic bytes for the serialized proof format.
private let kMagicBytes: [UInt8] = [0x5A, 0x4B, 0x53, 0x5A] // "ZKSZ"

/// Current format version.
private let kFormatVersion: UInt16 = 1

/// Header size in bytes.
private let kHeaderSize = 20

// MARK: - Compression Flags

/// Flags controlling serialization behavior.
public struct SerializerFlags: OptionSet {
    public let rawValue: UInt8

    public init(rawValue: UInt8) { self.rawValue = rawValue }

    /// Use compressed point representation (33 bytes vs 96 bytes for G1).
    public static let compressPoints  = SerializerFlags(rawValue: 1 << 0)
    /// Enable run-length encoding for repeated field elements.
    public static let runLengthEncode = SerializerFlags(rawValue: 1 << 1)
    /// Deduplicate identical points/elements via a reference table.
    public static let deduplication   = SerializerFlags(rawValue: 1 << 2)
    /// Strip identity points (encode as single zero byte).
    public static let stripIdentity   = SerializerFlags(rawValue: 1 << 3)

    /// Default: compress points + strip identity.
    public static let `default`: SerializerFlags = [.compressPoints, .stripIdentity]
    /// Maximum compression: all flags enabled.
    public static let maxCompression: SerializerFlags = [.compressPoints, .runLengthEncode, .deduplication, .stripIdentity]
    /// No compression: raw binary output.
    public static let none: SerializerFlags = []
}

// MARK: - Section Tags

/// Tags identifying sections within the serialized proof body.
public enum SectionTag: UInt8 {
    case fieldElements  = 0x01
    case groupPoints    = 0x02
    case merklePath     = 0x03
    case publicInputs   = 0x04
    case metadata       = 0x05
    case commitments    = 0x06
    case evaluations    = 0x07
    case friLayers      = 0x08
    case permutation    = 0x09
    case lookupData     = 0x0A
    case customData     = 0xFF
}

// MARK: - Serialized Proof Container

/// Container holding a fully serialized proof with metadata.
public struct SerializedProof {
    /// The raw serialized bytes (header + body).
    public let data: [UInt8]
    /// Proof system type.
    public let proofType: ProofSystemType
    /// Curve identifier.
    public let curveId: CurveId
    /// Uncompressed size in bytes (for compression ratio calculation).
    public let uncompressedSize: Int
    /// Flags used during serialization.
    public let flags: SerializerFlags

    /// Compression ratio (1.0 = no compression, lower = better).
    public var compressionRatio: Double {
        guard uncompressedSize > 0 else { return 1.0 }
        return Double(data.count) / Double(uncompressedSize)
    }
}

// MARK: - Size Estimator

/// Estimates serialized proof size without performing actual serialization.
public struct ProofSizeEstimator {

    /// Estimate the serialized size for a Groth16 proof.
    /// Groth16: 3 G1 points (A, B_G1 proxy, C) + public inputs.
    public static func estimateGroth16(publicInputCount: Int, flags: SerializerFlags = .default) -> Int {
        let pointSize = flags.contains(.compressPoints) ? 33 : 96
        let headerAndTags = kHeaderSize + 2 * (1 + 4) // 2 sections with tag + length
        let proofPoints = 3 * pointSize
        let pubInputs = publicInputCount * 32
        return headerAndTags + proofPoints + pubInputs
    }

    /// Estimate the serialized size for a Plonk proof.
    /// Plonk: wire commitments + grand product + quotient + opening evaluations + opening proofs.
    public static func estimatePlonk(numWires: Int, numEvaluations: Int,
                                     flags: SerializerFlags = .default) -> Int {
        let pointSize = flags.contains(.compressPoints) ? 33 : 96
        // wire commitments + grand product + quotient split + 2 opening proofs
        let numPoints = numWires + 1 + numWires + 2
        let headerAndTags = kHeaderSize + 3 * (1 + 4)
        return headerAndTags + numPoints * pointSize + numEvaluations * 32
    }

    /// Estimate the serialized size for a STARK proof.
    /// STARK: trace commitments (Merkle roots) + constraint evals + FRI layers.
    public static func estimateSTARK(traceWidth: Int, numFRILayers: Int,
                                     friLayerSize: Int, numQueries: Int,
                                     merkleDepth: Int) -> Int {
        let headerAndTags = kHeaderSize + 4 * (1 + 4)
        let traceRoots = traceWidth * 32
        let friData = numFRILayers * friLayerSize * 32
        let queryPaths = numQueries * merkleDepth * 32
        return headerAndTags + traceRoots + friData + queryPaths
    }

    /// Estimate the serialized size for a FRI proof.
    public static func estimateFRI(numLayers: Int, layerSize: Int,
                                   numQueries: Int, merkleDepth: Int) -> Int {
        let headerAndTags = kHeaderSize + 3 * (1 + 4)
        let layers = numLayers * layerSize * 32
        let queries = numQueries * merkleDepth * 32
        let finalPoly = layerSize * 32
        return headerAndTags + layers + queries + finalPoly
    }
}

// MARK: - Streaming Serialization Buffer

/// A streaming buffer that accumulates serialized data progressively.
/// Supports writing in chunks without requiring the full proof upfront.
public final class StreamingSerializationBuffer {
    private var buffer: [UInt8]
    private var sectionStart: Int?
    private var sectionTag: UInt8?
    private(set) public var bytesWritten: Int = 0

    /// Create a streaming buffer with an optional capacity hint.
    public init(capacity: Int = 4096) {
        buffer = []
        buffer.reserveCapacity(capacity)
    }

    /// Begin a new tagged section. Must call `endSection()` before starting another.
    public func beginSection(_ tag: SectionTag) {
        precondition(sectionStart == nil, "Must end current section before beginning a new one")
        sectionTag = tag.rawValue
        buffer.append(tag.rawValue)
        // Reserve 4 bytes for section length (will be patched in endSection)
        sectionStart = buffer.count
        buffer.append(contentsOf: [0, 0, 0, 0])
    }

    /// End the current section, patching in the section length.
    public func endSection() {
        guard let start = sectionStart else {
            preconditionFailure("No section in progress")
        }
        let sectionDataLength = buffer.count - start - 4
        let len = UInt32(sectionDataLength)
        buffer[start]     = UInt8(len & 0xFF)
        buffer[start + 1] = UInt8((len >> 8) & 0xFF)
        buffer[start + 2] = UInt8((len >> 16) & 0xFF)
        buffer[start + 3] = UInt8((len >> 24) & 0xFF)
        bytesWritten += 1 + 4 + sectionDataLength // tag + length + data
        sectionStart = nil
        sectionTag = nil
    }

    /// Write a field element into the current section.
    public func writeFr(_ value: Fr) {
        let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                     value.v.4, value.v.5, value.v.6, value.v.7]
        for limb in limbs {
            buffer.append(UInt8(limb & 0xFF))
            buffer.append(UInt8((limb >> 8) & 0xFF))
            buffer.append(UInt8((limb >> 16) & 0xFF))
            buffer.append(UInt8((limb >> 24) & 0xFF))
        }
    }

    /// Write a compressed G1 point (33 bytes) into the current section.
    public func writeCompressedPoint(_ p: PointProjective) {
        let compressed = bn254G1Compress(p)
        buffer.append(contentsOf: compressed)
    }

    /// Write a full projective point (96 bytes) into the current section.
    public func writeRawPoint(_ p: PointProjective) {
        writeFp(p.x)
        writeFp(p.y)
        writeFp(p.z)
    }

    /// Write an Fp element (32 bytes).
    public func writeFp(_ value: Fp) {
        let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                     value.v.4, value.v.5, value.v.6, value.v.7]
        for limb in limbs {
            buffer.append(UInt8(limb & 0xFF))
            buffer.append(UInt8((limb >> 8) & 0xFF))
            buffer.append(UInt8((limb >> 16) & 0xFF))
            buffer.append(UInt8((limb >> 24) & 0xFF))
        }
    }

    /// Write a UInt32 (little-endian).
    public func writeUInt32(_ value: UInt32) {
        buffer.append(UInt8(value & 0xFF))
        buffer.append(UInt8((value >> 8) & 0xFF))
        buffer.append(UInt8((value >> 16) & 0xFF))
        buffer.append(UInt8((value >> 24) & 0xFF))
    }

    /// Write raw bytes.
    public func writeBytes(_ bytes: [UInt8]) {
        buffer.append(contentsOf: bytes)
    }

    /// Write a Merkle path (array of 32-byte hashes).
    public func writeMerklePath(_ hashes: [[UInt8]]) {
        writeUInt32(UInt32(hashes.count))
        for h in hashes {
            precondition(h.count == 32, "Merkle hash must be 32 bytes")
            buffer.append(contentsOf: h)
        }
    }

    /// Get the accumulated buffer contents.
    public func finalize() -> [UInt8] { buffer }

    /// Current buffer size.
    public var size: Int { buffer.count }
}

// MARK: - Deserialization Reader

/// Reads sections from a serialized proof buffer.
public final class SerializedProofReader {
    private let data: [UInt8]
    private var offset: Int

    /// The proof system type from the header.
    public let proofType: ProofSystemType
    /// The curve ID from the header.
    public let curveId: CurveId
    /// Compression flags from the header.
    public let flags: SerializerFlags
    /// Format version from the header.
    public let formatVersion: UInt16
    /// Uncompressed size from the header.
    public let uncompressedSize: UInt32
    /// Compressed size from the header.
    public let compressedSize: UInt32

    /// Parse the header and prepare for section reading.
    /// Throws ProofSerializationError on invalid data.
    public init(_ data: [UInt8]) throws {
        guard data.count >= kHeaderSize else {
            throw ProofSerializationError.truncatedData(expected: kHeaderSize, available: data.count)
        }
        // Verify magic
        guard data[0] == kMagicBytes[0] && data[1] == kMagicBytes[1] &&
              data[2] == kMagicBytes[2] && data[3] == kMagicBytes[3] else {
            throw ProofSerializationError.wrongLabel(expected: "ZKSZ", got: "invalid magic")
        }
        self.formatVersion = UInt16(data[4]) | (UInt16(data[5]) << 8)
        guard let pt = ProofSystemType(rawValue: data[6]) else {
            throw ProofSerializationError.wrongLabel(expected: "valid proof type", got: "type \(data[6])")
        }
        self.proofType = pt
        guard let ci = CurveId(rawValue: data[7]) else {
            throw ProofSerializationError.wrongLabel(expected: "valid curve id", got: "curve \(data[7])")
        }
        self.curveId = ci
        self.flags = SerializerFlags(rawValue: data[8])
        self.uncompressedSize = UInt32(data[9]) | (UInt32(data[10]) << 8) |
                                (UInt32(data[11]) << 16) | (UInt32(data[12]) << 24)
        self.compressedSize = UInt32(data[13]) | (UInt32(data[14]) << 8) |
                              (UInt32(data[15]) << 16) | (UInt32(data[16]) << 24)
        self.data = data
        self.offset = kHeaderSize
    }

    /// Read the next section tag and its data. Returns nil at end of data.
    public func readSection() throws -> (tag: SectionTag, data: [UInt8])? {
        guard offset < data.count else { return nil }
        guard offset + 5 <= data.count else {
            throw ProofSerializationError.truncatedData(expected: 5, available: data.count - offset)
        }
        guard let tag = SectionTag(rawValue: data[offset]) else {
            throw ProofSerializationError.wrongLabel(expected: "valid section tag", got: "0x\(String(data[offset], radix: 16))")
        }
        offset += 1
        let length = Int(UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) |
                        (UInt32(data[offset + 2]) << 16) | (UInt32(data[offset + 3]) << 24))
        offset += 4
        guard offset + length <= data.count else {
            throw ProofSerializationError.truncatedData(expected: length, available: data.count - offset)
        }
        let sectionData = Array(data[offset..<(offset + length)])
        offset += length
        return (tag, sectionData)
    }

    /// Whether there are more sections to read.
    public var hasMoreSections: Bool { offset < data.count }

    /// Current read position.
    public var position: Int { offset }

    // MARK: - Helpers for parsing section data

    /// Read an Fr from a section data buffer at a given offset.
    public static func readFr(from buf: [UInt8], at pos: inout Int) throws -> Fr {
        guard pos + 32 <= buf.count else {
            throw ProofSerializationError.truncatedData(expected: 32, available: buf.count - pos)
        }
        var limbs = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            let b = pos + i * 4
            limbs[i] = UInt32(buf[b]) | (UInt32(buf[b + 1]) << 8) |
                       (UInt32(buf[b + 2]) << 16) | (UInt32(buf[b + 3]) << 24)
        }
        pos += 32
        return Fr(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                      limbs[4], limbs[5], limbs[6], limbs[7]))
    }

    /// Read a compressed G1 point (33 bytes) from a section data buffer.
    public static func readCompressedPoint(from buf: [UInt8], at pos: inout Int) throws -> PointProjective {
        guard pos + 33 <= buf.count else {
            throw ProofSerializationError.truncatedData(expected: 33, available: buf.count - pos)
        }
        let pointData = Array(buf[pos..<(pos + 33)])
        pos += 33
        guard let p = bn254G1Decompress(pointData) else {
            throw ProofSerializationError.wrongLabel(expected: "valid compressed point", got: "decompression failed")
        }
        return p
    }

    /// Read a raw projective point (96 bytes) from a section data buffer.
    public static func readRawPoint(from buf: [UInt8], at pos: inout Int) throws -> PointProjective {
        guard pos + 96 <= buf.count else {
            throw ProofSerializationError.truncatedData(expected: 96, available: buf.count - pos)
        }
        var limbs = [UInt32](repeating: 0, count: 8)
        func readFpAt(_ base: Int) -> Fp {
            for i in 0..<8 {
                let b = base + i * 4
                limbs[i] = UInt32(buf[b]) | (UInt32(buf[b + 1]) << 8) |
                           (UInt32(buf[b + 2]) << 16) | (UInt32(buf[b + 3]) << 24)
            }
            return Fp(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                          limbs[4], limbs[5], limbs[6], limbs[7]))
        }
        let x = readFpAt(pos)
        let y = readFpAt(pos + 32)
        let z = readFpAt(pos + 64)
        pos += 96
        return PointProjective(x: x, y: y, z: z)
    }

    /// Read a UInt32 from a section data buffer.
    public static func readUInt32(from buf: [UInt8], at pos: inout Int) throws -> UInt32 {
        guard pos + 4 <= buf.count else {
            throw ProofSerializationError.truncatedData(expected: 4, available: buf.count - pos)
        }
        let val = UInt32(buf[pos]) | (UInt32(buf[pos + 1]) << 8) |
                  (UInt32(buf[pos + 2]) << 16) | (UInt32(buf[pos + 3]) << 24)
        pos += 4
        return val
    }

    /// Read a Merkle path from a section data buffer.
    public static func readMerklePath(from buf: [UInt8], at pos: inout Int) throws -> [[UInt8]] {
        let count = Int(try readUInt32(from: buf, at: &pos))
        var result = [[UInt8]]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            guard pos + 32 <= buf.count else {
                throw ProofSerializationError.truncatedData(expected: 32, available: buf.count - pos)
            }
            result.append(Array(buf[pos..<(pos + 32)]))
            pos += 32
        }
        return result
    }
}

// MARK: - Run-Length Encoding for Field Elements

/// Run-length encoder for sequences of field elements.
/// Encodes runs of identical Fr values as (count, value) pairs.
private struct FrRunLengthEncoder {
    /// Encode a sequence of Fr values with run-length encoding.
    /// Format: [numRuns: u32] then for each run: [count: u32][value: 32 bytes]
    static func encode(_ elements: [Fr]) -> [UInt8] {
        guard !elements.isEmpty else {
            var out = [UInt8](repeating: 0, count: 4)
            return out // numRuns = 0
        }

        var runs: [(count: UInt32, value: Fr)] = []
        var current = elements[0]
        var count: UInt32 = 1

        for i in 1..<elements.count {
            if frEqualRaw(elements[i], current) {
                count += 1
            } else {
                runs.append((count, current))
                current = elements[i]
                count = 1
            }
        }
        runs.append((count, current))

        // Serialize
        var out = [UInt8]()
        out.reserveCapacity(4 + runs.count * 36)
        appendUInt32(&out, UInt32(runs.count))
        for run in runs {
            appendUInt32(&out, run.count)
            appendFr(&out, run.value)
        }
        return out
    }

    /// Decode a run-length encoded field element sequence.
    static func decode(_ data: [UInt8]) throws -> [Fr] {
        var pos = 0
        let numRuns = Int(try readU32(data, &pos))
        var result = [Fr]()
        for _ in 0..<numRuns {
            let count = Int(try readU32(data, &pos))
            let value = try readFrVal(data, &pos)
            for _ in 0..<count {
                result.append(value)
            }
        }
        return result
    }

    private static func readU32(_ data: [UInt8], _ pos: inout Int) throws -> UInt32 {
        guard pos + 4 <= data.count else {
            throw ProofSerializationError.truncatedData(expected: 4, available: data.count - pos)
        }
        let val = UInt32(data[pos]) | (UInt32(data[pos + 1]) << 8) |
                  (UInt32(data[pos + 2]) << 16) | (UInt32(data[pos + 3]) << 24)
        pos += 4
        return val
    }

    private static func readFrVal(_ data: [UInt8], _ pos: inout Int) throws -> Fr {
        guard pos + 32 <= data.count else {
            throw ProofSerializationError.truncatedData(expected: 32, available: data.count - pos)
        }
        var limbs = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            let b = pos + i * 4
            limbs[i] = UInt32(data[b]) | (UInt32(data[b + 1]) << 8) |
                       (UInt32(data[b + 2]) << 16) | (UInt32(data[b + 3]) << 24)
        }
        pos += 32
        return Fr(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                      limbs[4], limbs[5], limbs[6], limbs[7]))
    }
}

// MARK: - Internal Helpers

/// Bitwise equality check for Fr values (no field reduction).
private func frEqualRaw(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func appendUInt32(_ buf: inout [UInt8], _ val: UInt32) {
    buf.append(UInt8(val & 0xFF))
    buf.append(UInt8((val >> 8) & 0xFF))
    buf.append(UInt8((val >> 16) & 0xFF))
    buf.append(UInt8((val >> 24) & 0xFF))
}

private func appendFr(_ buf: inout [UInt8], _ value: Fr) {
    let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                 value.v.4, value.v.5, value.v.6, value.v.7]
    for limb in limbs {
        buf.append(UInt8(limb & 0xFF))
        buf.append(UInt8((limb >> 8) & 0xFF))
        buf.append(UInt8((limb >> 16) & 0xFF))
        buf.append(UInt8((limb >> 24) & 0xFF))
    }
}

// MARK: - GPU Proof Serializer Engine

/// GPU-accelerated proof serialization engine.
///
/// Provides efficient serialization and deserialization of zero-knowledge proofs
/// in a compact binary format with versioning, compression, and streaming support.
///
/// Supported proof systems:
///   - Groth16: 3 G1 curve points + public inputs
///   - Plonk: wire commitments + evaluations + opening proofs
///   - STARK: trace commitments + FRI layers + query paths
///   - FRI: folded layers + query Merkle paths + final polynomial
///
/// Usage:
/// ```swift
/// let engine = GPUProofSerializerEngine()
///
/// // Serialize a Groth16 proof
/// let serialized = engine.serializeGroth16(a: proofA, b: proofB, c: proofC,
///                                          publicInputs: pubInputs)
///
/// // Deserialize
/// let (a, b, c, pubIn) = try engine.deserializeGroth16(serialized)
///
/// // Estimate size before serializing
/// let estimated = ProofSizeEstimator.estimateGroth16(publicInputCount: 3)
/// ```
public final class GPUProofSerializerEngine {

    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Compression flags for this engine instance.
    public let flags: SerializerFlags

    /// Create a serializer engine with the given compression flags.
    public init(flags: SerializerFlags = .default) {
        self.flags = flags
    }

    // MARK: - Header Construction

    /// Build the 20-byte header for a serialized proof.
    private func buildHeader(proofType: ProofSystemType, curveId: CurveId,
                             uncompressedSize: UInt32, compressedSize: UInt32) -> [UInt8] {
        var header = [UInt8](repeating: 0, count: kHeaderSize)
        // Magic
        header[0] = kMagicBytes[0]
        header[1] = kMagicBytes[1]
        header[2] = kMagicBytes[2]
        header[3] = kMagicBytes[3]
        // Version
        header[4] = UInt8(kFormatVersion & 0xFF)
        header[5] = UInt8((kFormatVersion >> 8) & 0xFF)
        // Proof type + curve
        header[6] = proofType.rawValue
        header[7] = curveId.rawValue
        // Flags
        header[8] = flags.rawValue
        // Uncompressed size
        header[9]  = UInt8(uncompressedSize & 0xFF)
        header[10] = UInt8((uncompressedSize >> 8) & 0xFF)
        header[11] = UInt8((uncompressedSize >> 16) & 0xFF)
        header[12] = UInt8((uncompressedSize >> 24) & 0xFF)
        // Compressed size
        header[13] = UInt8(compressedSize & 0xFF)
        header[14] = UInt8((compressedSize >> 8) & 0xFF)
        header[15] = UInt8((compressedSize >> 16) & 0xFF)
        header[16] = UInt8((compressedSize >> 24) & 0xFF)
        // Reserved [17..19] = 0
        return header
    }

    // MARK: - Point Serialization

    /// Serialize a point based on the engine's compression flags.
    private func serializePoint(_ p: PointProjective, into buf: inout [UInt8]) {
        if flags.contains(.stripIdentity) && pointIsIdentity(p) {
            buf.append(0x00) // Single zero byte for identity
            return
        }
        if flags.contains(.compressPoints) {
            let compressed = bn254G1Compress(p)
            buf.append(contentsOf: compressed)
        } else {
            // Raw projective: 96 bytes
            appendFp(&buf, p.x)
            appendFp(&buf, p.y)
            appendFp(&buf, p.z)
        }
    }

    /// Deserialize a point based on the engine's compression flags.
    private func deserializePoint(from buf: [UInt8], at pos: inout Int) throws -> PointProjective {
        guard pos < buf.count else {
            throw ProofSerializationError.truncatedData(expected: 1, available: 0)
        }
        if flags.contains(.stripIdentity) && buf[pos] == 0x00 {
            // Check if this is truly a stripped identity (single zero byte)
            // vs a compressed point starting with 0x00 (which is also identity in SEC1)
            if flags.contains(.compressPoints) {
                // In compressed mode, identity is 33 zero bytes or our single 0x00
                // We use single 0x00 for stripped identity
                pos += 1
                return pointIdentity()
            } else {
                pos += 1
                return pointIdentity()
            }
        }
        if flags.contains(.compressPoints) {
            guard pos + 33 <= buf.count else {
                throw ProofSerializationError.truncatedData(expected: 33, available: buf.count - pos)
            }
            let pointData = Array(buf[pos..<(pos + 33)])
            pos += 33
            guard let p = bn254G1Decompress(pointData) else {
                throw ProofSerializationError.wrongLabel(expected: "valid compressed point", got: "decompression failed")
            }
            return p
        } else {
            return try SerializedProofReader.readRawPoint(from: buf, at: &pos)
        }
    }

    private func appendFp(_ buf: inout [UInt8], _ value: Fp) {
        let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                     value.v.4, value.v.5, value.v.6, value.v.7]
        for limb in limbs {
            buf.append(UInt8(limb & 0xFF))
            buf.append(UInt8((limb >> 8) & 0xFF))
            buf.append(UInt8((limb >> 16) & 0xFF))
            buf.append(UInt8((limb >> 24) & 0xFF))
        }
    }

    // MARK: - Groth16 Serialization

    /// Serialize a Groth16 proof (3 G1 points + public inputs).
    ///
    /// - Parameters:
    ///   - a: Proof point A (G1)
    ///   - b: Proof point B (G1, proxy for the G2 element in pairing)
    ///   - c: Proof point C (G1)
    ///   - publicInputs: Public input field elements
    /// - Returns: Serialized proof container
    public func serializeGroth16(a: PointProjective, b: PointProjective,
                                 c: PointProjective,
                                 publicInputs: [Fr]) -> SerializedProof {
        // Build body: points section + public inputs section
        var body = [UInt8]()
        let uncompressedSize = 3 * 96 + publicInputs.count * 32

        // Section: group points (A, B, C)
        body.append(SectionTag.groupPoints.rawValue)
        var pointsData = [UInt8]()
        appendUInt32(&pointsData, 3) // 3 points
        serializePoint(a, into: &pointsData)
        serializePoint(b, into: &pointsData)
        serializePoint(c, into: &pointsData)
        appendUInt32(&body, UInt32(pointsData.count))
        body.append(contentsOf: pointsData)

        // Section: public inputs
        body.append(SectionTag.publicInputs.rawValue)
        var pubData = [UInt8]()
        if flags.contains(.runLengthEncode) {
            pubData = FrRunLengthEncoder.encode(publicInputs)
        } else {
            appendUInt32(&pubData, UInt32(publicInputs.count))
            for v in publicInputs { appendFr(&pubData, v) }
        }
        appendUInt32(&body, UInt32(pubData.count))
        body.append(contentsOf: pubData)

        let header = buildHeader(proofType: .groth16, curveId: .bn254,
                                 uncompressedSize: UInt32(uncompressedSize),
                                 compressedSize: UInt32(body.count))
        var result = header
        result.append(contentsOf: body)

        return SerializedProof(data: result, proofType: .groth16, curveId: .bn254,
                               uncompressedSize: uncompressedSize, flags: flags)
    }

    /// Deserialize a Groth16 proof.
    /// - Returns: (A, B, C, publicInputs)
    public func deserializeGroth16(_ serialized: SerializedProof) throws
        -> (a: PointProjective, b: PointProjective, c: PointProjective, publicInputs: [Fr]) {
        let reader = try SerializedProofReader(serialized.data)
        guard reader.proofType == .groth16 else {
            throw ProofSerializationError.wrongLabel(expected: "groth16", got: reader.proofType.description)
        }

        var a = pointIdentity()
        var b = pointIdentity()
        var c = pointIdentity()
        var publicInputs = [Fr]()

        while reader.hasMoreSections {
            guard let (tag, sectionData) = try reader.readSection() else { break }
            switch tag {
            case .groupPoints:
                var pos = 0
                let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                guard count >= 3 else {
                    throw ProofSerializationError.truncatedData(expected: 3, available: count)
                }
                a = try deserializePoint(from: sectionData, at: &pos)
                b = try deserializePoint(from: sectionData, at: &pos)
                c = try deserializePoint(from: sectionData, at: &pos)
            case .publicInputs:
                if flags.contains(.runLengthEncode) {
                    publicInputs = try FrRunLengthEncoder.decode(sectionData)
                } else {
                    var pos = 0
                    let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                    publicInputs.reserveCapacity(count)
                    for _ in 0..<count {
                        publicInputs.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                    }
                }
            default:
                break // Skip unknown sections for forward compatibility
            }
        }
        return (a, b, c, publicInputs)
    }

    // MARK: - Plonk Serialization

    /// Serialize a Plonk proof.
    ///
    /// - Parameters:
    ///   - wireCommitments: Wire polynomial commitments (G1 points)
    ///   - grandProductCommitment: Grand product commitment (G1)
    ///   - quotientCommitments: Quotient polynomial commitments (G1 points)
    ///   - openingProof: Opening proof point (G1)
    ///   - shiftedOpeningProof: Shifted opening proof point (G1)
    ///   - evaluations: Proof evaluations (Fr elements)
    /// - Returns: Serialized proof container
    public func serializePlonk(wireCommitments: [PointProjective],
                               grandProductCommitment: PointProjective,
                               quotientCommitments: [PointProjective],
                               openingProof: PointProjective,
                               shiftedOpeningProof: PointProjective,
                               evaluations: [Fr]) -> SerializedProof {
        var body = [UInt8]()
        let numPoints = wireCommitments.count + 1 + quotientCommitments.count + 2
        let uncompressedSize = numPoints * 96 + evaluations.count * 32

        // Section: commitments
        body.append(SectionTag.commitments.rawValue)
        var commitData = [UInt8]()
        // Encode counts for structured deserialization
        appendUInt32(&commitData, UInt32(wireCommitments.count))
        for w in wireCommitments { serializePoint(w, into: &commitData) }
        serializePoint(grandProductCommitment, into: &commitData)
        appendUInt32(&commitData, UInt32(quotientCommitments.count))
        for q in quotientCommitments { serializePoint(q, into: &commitData) }
        serializePoint(openingProof, into: &commitData)
        serializePoint(shiftedOpeningProof, into: &commitData)
        appendUInt32(&body, UInt32(commitData.count))
        body.append(contentsOf: commitData)

        // Section: evaluations
        body.append(SectionTag.evaluations.rawValue)
        var evalData = [UInt8]()
        if flags.contains(.runLengthEncode) {
            evalData = FrRunLengthEncoder.encode(evaluations)
        } else {
            appendUInt32(&evalData, UInt32(evaluations.count))
            for e in evaluations { appendFr(&evalData, e) }
        }
        appendUInt32(&body, UInt32(evalData.count))
        body.append(contentsOf: evalData)

        let header = buildHeader(proofType: .plonk, curveId: .bn254,
                                 uncompressedSize: UInt32(uncompressedSize),
                                 compressedSize: UInt32(body.count))
        var result = header
        result.append(contentsOf: body)

        return SerializedProof(data: result, proofType: .plonk, curveId: .bn254,
                               uncompressedSize: uncompressedSize, flags: flags)
    }

    /// Deserialize a Plonk proof.
    public func deserializePlonk(_ serialized: SerializedProof) throws
        -> (wireCommitments: [PointProjective], grandProductCommitment: PointProjective,
            quotientCommitments: [PointProjective], openingProof: PointProjective,
            shiftedOpeningProof: PointProjective, evaluations: [Fr]) {
        let reader = try SerializedProofReader(serialized.data)
        guard reader.proofType == .plonk else {
            throw ProofSerializationError.wrongLabel(expected: "plonk", got: reader.proofType.description)
        }

        var wireCommitments = [PointProjective]()
        var grandProduct = pointIdentity()
        var quotientCommitments = [PointProjective]()
        var openingProof = pointIdentity()
        var shiftedOpeningProof = pointIdentity()
        var evaluations = [Fr]()

        while reader.hasMoreSections {
            guard let (tag, sectionData) = try reader.readSection() else { break }
            switch tag {
            case .commitments:
                var pos = 0
                let wireCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                wireCommitments.reserveCapacity(wireCount)
                for _ in 0..<wireCount {
                    wireCommitments.append(try deserializePoint(from: sectionData, at: &pos))
                }
                grandProduct = try deserializePoint(from: sectionData, at: &pos)
                let quotientCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                quotientCommitments.reserveCapacity(quotientCount)
                for _ in 0..<quotientCount {
                    quotientCommitments.append(try deserializePoint(from: sectionData, at: &pos))
                }
                openingProof = try deserializePoint(from: sectionData, at: &pos)
                shiftedOpeningProof = try deserializePoint(from: sectionData, at: &pos)
            case .evaluations:
                if flags.contains(.runLengthEncode) {
                    evaluations = try FrRunLengthEncoder.decode(sectionData)
                } else {
                    var pos = 0
                    let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                    evaluations.reserveCapacity(count)
                    for _ in 0..<count {
                        evaluations.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                    }
                }
            default:
                break
            }
        }
        return (wireCommitments, grandProduct, quotientCommitments, openingProof, shiftedOpeningProof, evaluations)
    }

    // MARK: - STARK Serialization

    /// Serialize a STARK proof.
    ///
    /// - Parameters:
    ///   - traceCommitments: Merkle roots of trace columns (32 bytes each)
    ///   - constraintEvaluations: Constraint evaluation field elements
    ///   - friLayers: FRI oracle layers (array of arrays of Fr)
    ///   - queryPaths: Merkle authentication paths for queries
    /// - Returns: Serialized proof container
    public func serializeSTARK(traceCommitments: [[UInt8]],
                               constraintEvaluations: [Fr],
                               friLayers: [[Fr]],
                               queryPaths: [[[UInt8]]]) -> SerializedProof {
        var body = [UInt8]()
        let uncompressedSize = traceCommitments.count * 32 +
            constraintEvaluations.count * 32 +
            friLayers.reduce(0) { $0 + $1.count * 32 } +
            queryPaths.reduce(0) { $0 + $1.reduce(0) { $0 + $1.count } }

        // Section: metadata (trace commitments / Merkle roots)
        body.append(SectionTag.metadata.rawValue)
        var metaData = [UInt8]()
        appendUInt32(&metaData, UInt32(traceCommitments.count))
        for root in traceCommitments {
            metaData.append(contentsOf: root)
        }
        appendUInt32(&body, UInt32(metaData.count))
        body.append(contentsOf: metaData)

        // Section: constraint evaluations
        body.append(SectionTag.evaluations.rawValue)
        var evalData = [UInt8]()
        if flags.contains(.runLengthEncode) {
            evalData = FrRunLengthEncoder.encode(constraintEvaluations)
        } else {
            appendUInt32(&evalData, UInt32(constraintEvaluations.count))
            for e in constraintEvaluations { appendFr(&evalData, e) }
        }
        appendUInt32(&body, UInt32(evalData.count))
        body.append(contentsOf: evalData)

        // Section: FRI layers
        body.append(SectionTag.friLayers.rawValue)
        var friData = [UInt8]()
        appendUInt32(&friData, UInt32(friLayers.count))
        for layer in friLayers {
            if flags.contains(.runLengthEncode) {
                let encoded = FrRunLengthEncoder.encode(layer)
                appendUInt32(&friData, UInt32(encoded.count))
                friData.append(contentsOf: encoded)
            } else {
                appendUInt32(&friData, UInt32(layer.count))
                for e in layer { appendFr(&friData, e) }
            }
        }
        appendUInt32(&body, UInt32(friData.count))
        body.append(contentsOf: friData)

        // Section: Merkle query paths
        body.append(SectionTag.merklePath.rawValue)
        var pathData = [UInt8]()
        appendUInt32(&pathData, UInt32(queryPaths.count))
        for path in queryPaths {
            appendUInt32(&pathData, UInt32(path.count))
            for hash in path {
                pathData.append(contentsOf: hash)
            }
        }
        appendUInt32(&body, UInt32(pathData.count))
        body.append(contentsOf: pathData)

        let header = buildHeader(proofType: .stark, curveId: .none,
                                 uncompressedSize: UInt32(uncompressedSize),
                                 compressedSize: UInt32(body.count))
        var result = header
        result.append(contentsOf: body)

        return SerializedProof(data: result, proofType: .stark, curveId: .none,
                               uncompressedSize: uncompressedSize, flags: flags)
    }

    /// Deserialize a STARK proof.
    public func deserializeSTARK(_ serialized: SerializedProof) throws
        -> (traceCommitments: [[UInt8]], constraintEvaluations: [Fr],
            friLayers: [[Fr]], queryPaths: [[[UInt8]]]) {
        let reader = try SerializedProofReader(serialized.data)
        guard reader.proofType == .stark else {
            throw ProofSerializationError.wrongLabel(expected: "stark", got: reader.proofType.description)
        }

        var traceCommitments = [[UInt8]]()
        var constraintEvals = [Fr]()
        var friLayers = [[Fr]]()
        var queryPaths = [[[UInt8]]]()

        while reader.hasMoreSections {
            guard let (tag, sectionData) = try reader.readSection() else { break }
            switch tag {
            case .metadata:
                var pos = 0
                let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                traceCommitments.reserveCapacity(count)
                for _ in 0..<count {
                    guard pos + 32 <= sectionData.count else {
                        throw ProofSerializationError.truncatedData(expected: 32, available: sectionData.count - pos)
                    }
                    traceCommitments.append(Array(sectionData[pos..<(pos + 32)]))
                    pos += 32
                }
            case .evaluations:
                if flags.contains(.runLengthEncode) {
                    constraintEvals = try FrRunLengthEncoder.decode(sectionData)
                } else {
                    var pos = 0
                    let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                    constraintEvals.reserveCapacity(count)
                    for _ in 0..<count {
                        constraintEvals.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                    }
                }
            case .friLayers:
                var pos = 0
                let layerCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                friLayers.reserveCapacity(layerCount)
                for _ in 0..<layerCount {
                    if flags.contains(.runLengthEncode) {
                        let encodedLen = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                        let encodedData = Array(sectionData[pos..<(pos + encodedLen)])
                        pos += encodedLen
                        friLayers.append(try FrRunLengthEncoder.decode(encodedData))
                    } else {
                        let elemCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                        var layer = [Fr]()
                        layer.reserveCapacity(elemCount)
                        for _ in 0..<elemCount {
                            layer.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                        }
                        friLayers.append(layer)
                    }
                }
            case .merklePath:
                var pos = 0
                let pathCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                queryPaths.reserveCapacity(pathCount)
                for _ in 0..<pathCount {
                    let hashCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                    var path = [[UInt8]]()
                    path.reserveCapacity(hashCount)
                    for _ in 0..<hashCount {
                        guard pos + 32 <= sectionData.count else {
                            throw ProofSerializationError.truncatedData(expected: 32, available: sectionData.count - pos)
                        }
                        path.append(Array(sectionData[pos..<(pos + 32)]))
                        pos += 32
                    }
                    queryPaths.append(path)
                }
            default:
                break
            }
        }
        return (traceCommitments, constraintEvals, friLayers, queryPaths)
    }

    // MARK: - FRI Serialization

    /// Serialize a FRI proof.
    ///
    /// - Parameters:
    ///   - layers: Folded FRI layers (array of arrays of Fr)
    ///   - queryPaths: Merkle authentication paths for queries
    ///   - finalPoly: Final constant polynomial coefficients
    /// - Returns: Serialized proof container
    public func serializeFRI(layers: [[Fr]], queryPaths: [[[UInt8]]],
                             finalPoly: [Fr]) -> SerializedProof {
        var body = [UInt8]()
        let uncompressedSize = layers.reduce(0) { $0 + $1.count * 32 } +
            queryPaths.reduce(0) { $0 + $1.reduce(0) { $0 + $1.count } } +
            finalPoly.count * 32

        // Section: FRI layers
        body.append(SectionTag.friLayers.rawValue)
        var friData = [UInt8]()
        appendUInt32(&friData, UInt32(layers.count))
        for layer in layers {
            if flags.contains(.runLengthEncode) {
                let encoded = FrRunLengthEncoder.encode(layer)
                appendUInt32(&friData, UInt32(encoded.count))
                friData.append(contentsOf: encoded)
            } else {
                appendUInt32(&friData, UInt32(layer.count))
                for e in layer { appendFr(&friData, e) }
            }
        }
        appendUInt32(&body, UInt32(friData.count))
        body.append(contentsOf: friData)

        // Section: Merkle query paths
        body.append(SectionTag.merklePath.rawValue)
        var pathData = [UInt8]()
        appendUInt32(&pathData, UInt32(queryPaths.count))
        for path in queryPaths {
            appendUInt32(&pathData, UInt32(path.count))
            for hash in path { pathData.append(contentsOf: hash) }
        }
        appendUInt32(&body, UInt32(pathData.count))
        body.append(contentsOf: pathData)

        // Section: final polynomial (as field elements)
        body.append(SectionTag.fieldElements.rawValue)
        var polyData = [UInt8]()
        appendUInt32(&polyData, UInt32(finalPoly.count))
        for e in finalPoly { appendFr(&polyData, e) }
        appendUInt32(&body, UInt32(polyData.count))
        body.append(contentsOf: polyData)

        let header = buildHeader(proofType: .fri, curveId: .none,
                                 uncompressedSize: UInt32(uncompressedSize),
                                 compressedSize: UInt32(body.count))
        var result = header
        result.append(contentsOf: body)

        return SerializedProof(data: result, proofType: .fri, curveId: .none,
                               uncompressedSize: uncompressedSize, flags: flags)
    }

    /// Deserialize a FRI proof.
    public func deserializeFRI(_ serialized: SerializedProof) throws
        -> (layers: [[Fr]], queryPaths: [[[UInt8]]], finalPoly: [Fr]) {
        let reader = try SerializedProofReader(serialized.data)
        guard reader.proofType == .fri else {
            throw ProofSerializationError.wrongLabel(expected: "fri", got: reader.proofType.description)
        }

        var layers = [[Fr]]()
        var queryPaths = [[[UInt8]]]()
        var finalPoly = [Fr]()

        while reader.hasMoreSections {
            guard let (tag, sectionData) = try reader.readSection() else { break }
            switch tag {
            case .friLayers:
                var pos = 0
                let layerCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                layers.reserveCapacity(layerCount)
                for _ in 0..<layerCount {
                    if flags.contains(.runLengthEncode) {
                        let encodedLen = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                        let encodedData = Array(sectionData[pos..<(pos + encodedLen)])
                        pos += encodedLen
                        layers.append(try FrRunLengthEncoder.decode(encodedData))
                    } else {
                        let elemCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                        var layer = [Fr]()
                        layer.reserveCapacity(elemCount)
                        for _ in 0..<elemCount {
                            layer.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                        }
                        layers.append(layer)
                    }
                }
            case .merklePath:
                var pos = 0
                let pathCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                queryPaths.reserveCapacity(pathCount)
                for _ in 0..<pathCount {
                    let hashCount = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                    var path = [[UInt8]]()
                    path.reserveCapacity(hashCount)
                    for _ in 0..<hashCount {
                        guard pos + 32 <= sectionData.count else {
                            throw ProofSerializationError.truncatedData(expected: 32, available: sectionData.count - pos)
                        }
                        path.append(Array(sectionData[pos..<(pos + 32)]))
                        pos += 32
                    }
                    queryPaths.append(path)
                }
            case .fieldElements:
                var pos = 0
                let count = Int(try SerializedProofReader.readUInt32(from: sectionData, at: &pos))
                finalPoly.reserveCapacity(count)
                for _ in 0..<count {
                    finalPoly.append(try SerializedProofReader.readFr(from: sectionData, at: &pos))
                }
            default:
                break
            }
        }
        return (layers, queryPaths, finalPoly)
    }

    // MARK: - Generic Proof Serialization (Streaming)

    /// Serialize arbitrary proof data using the streaming buffer.
    ///
    /// This method provides maximum flexibility for custom proof formats.
    /// The caller provides a closure that writes sections into the streaming buffer.
    ///
    /// - Parameters:
    ///   - proofType: The proof system type
    ///   - curveId: The curve identifier
    ///   - estimatedSize: Estimated uncompressed size (for header metadata)
    ///   - writer: Closure that writes proof data into the streaming buffer
    /// - Returns: Serialized proof container
    public func serializeCustom(proofType: ProofSystemType, curveId: CurveId,
                                estimatedSize: Int,
                                writer: (StreamingSerializationBuffer) -> Void) -> SerializedProof {
        let stream = StreamingSerializationBuffer(capacity: estimatedSize)
        writer(stream)
        let body = stream.finalize()

        let header = buildHeader(proofType: proofType, curveId: curveId,
                                 uncompressedSize: UInt32(estimatedSize),
                                 compressedSize: UInt32(body.count))
        var result = header
        result.append(contentsOf: body)

        return SerializedProof(data: result, proofType: proofType, curveId: curveId,
                               uncompressedSize: estimatedSize, flags: flags)
    }

    // MARK: - Proof Type Detection

    /// Detect the proof system type from raw serialized bytes without full deserialization.
    /// Returns nil if the data is not a valid serialized proof.
    public static func detectProofType(_ data: [UInt8]) -> ProofSystemType? {
        guard data.count >= kHeaderSize else { return nil }
        guard data[0] == kMagicBytes[0] && data[1] == kMagicBytes[1] &&
              data[2] == kMagicBytes[2] && data[3] == kMagicBytes[3] else { return nil }
        return ProofSystemType(rawValue: data[6])
    }

    /// Detect the curve ID from raw serialized bytes.
    public static func detectCurveId(_ data: [UInt8]) -> CurveId? {
        guard data.count >= kHeaderSize else { return nil }
        guard data[0] == kMagicBytes[0] && data[1] == kMagicBytes[1] &&
              data[2] == kMagicBytes[2] && data[3] == kMagicBytes[3] else { return nil }
        return CurveId(rawValue: data[7])
    }

    /// Check format version compatibility.
    public static func isCompatible(_ data: [UInt8]) -> Bool {
        guard data.count >= kHeaderSize else { return false }
        guard data[0] == kMagicBytes[0] && data[1] == kMagicBytes[1] &&
              data[2] == kMagicBytes[2] && data[3] == kMagicBytes[3] else { return false }
        let version = UInt16(data[4]) | (UInt16(data[5]) << 8)
        return version <= kFormatVersion
    }
}
