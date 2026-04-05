// SRS File Format — Read/write KZG trusted setup files.
//
// Supported formats:
//   - Hermez .ptau (powers of tau, used by snarkjs/circom)
//   - Ethereum KZG ceremony (EIP-4844 trusted setup)
//
// Both formats store G1 and G2 points from a powers-of-tau ceremony.
// This module handles serialization/deserialization and format conversion.

import Foundation

// MARK: - File Format Enum

/// Supported SRS file formats.
public enum SRSFileFormat {
    /// Hermez/snarkjs .ptau format (BN254 or BLS12-381).
    /// Header: magic + version + sections (G1, G2, contributions metadata).
    case ptau

    /// Ethereum KZG ceremony format (BLS12-381 only).
    /// Simple concatenation: N compressed G1 points (48 bytes each) +
    /// M compressed G2 points (96 bytes each).
    case ethereumKZG
}

// MARK: - Load SRS

/// Load an SRS from file data in the specified format.
///
/// - Parameters:
///   - data: Raw file bytes.
///   - format: Which file format to parse.
/// - Returns: The parsed SRS, or nil if the data is invalid.
public func loadSRS(from data: [UInt8], format: SRSFileFormat) -> StructuredReferenceString? {
    switch format {
    case .ptau:
        return loadPtau(data)
    case .ethereumKZG:
        return loadEthereumKZG(data)
    }
}

/// Save an SRS to file data in the specified format.
///
/// - Parameters:
///   - srs: The SRS to serialize.
///   - format: Which file format to write.
/// - Returns: The serialized bytes, or nil if the format is incompatible.
public func saveSRS(_ srs: StructuredReferenceString, format: SRSFileFormat) -> [UInt8]? {
    switch format {
    case .ptau:
        return savePtau(srs)
    case .ethereumKZG:
        return saveEthereumKZG(srs)
    }
}

// MARK: - Hermez .ptau Format

// .ptau file layout:
//   Header:
//     4 bytes: magic "ptau"
//     4 bytes: version (LE uint32, currently 1)
//     4 bytes: number of sections
//   Sections (each):
//     4 bytes: section type
//     8 bytes: section length (LE uint64)
//     N bytes: section data
//
//   Section types:
//     1 = Header (curve ID, power, ceremony power)
//     2 = G1 tau powers (uncompressed affine points)
//     3 = G2 tau powers (uncompressed affine points)
//     4 = G1 alpha*tau powers
//     5 = G1 beta*tau powers
//     6 = beta*G2
//     7 = Contributions
//
//   We only need sections 1, 2, 3 for basic SRS loading.
//   Curve IDs: 1 = BN254, 2 = BLS12-381

private let PTAU_MAGIC: [UInt8] = [0x70, 0x74, 0x61, 0x75]  // "ptau"
private let PTAU_VERSION: UInt32 = 1

private let PTAU_SECTION_HEADER: UInt32 = 1
private let PTAU_SECTION_G1_TAU: UInt32 = 2
private let PTAU_SECTION_G2_TAU: UInt32 = 3

private let PTAU_CURVE_BN254: UInt32 = 1
private let PTAU_CURVE_BLS12381: UInt32 = 2

private func loadPtau(_ data: [UInt8]) -> StructuredReferenceString? {
    guard data.count >= 12 else { return nil }

    // Check magic
    guard Array(data[0..<4]) == PTAU_MAGIC else { return nil }

    // Version
    let version = readUInt32LE(data, offset: 4)
    guard version == PTAU_VERSION else { return nil }

    // Number of sections
    let numSections = readUInt32LE(data, offset: 8)

    // Parse sections
    var offset = 12
    var headerData: [UInt8]?
    var g1Data: [UInt8]?
    var g2Data: [UInt8]?

    for _ in 0..<numSections {
        guard offset + 12 <= data.count else { return nil }
        let sectionType = readUInt32LE(data, offset: offset)
        let sectionLen = readUInt64LE(data, offset: offset + 4)
        offset += 12
        let sectionEnd = offset + Int(sectionLen)
        guard sectionEnd <= data.count else { return nil }

        let sectionBytes = Array(data[offset..<sectionEnd])
        switch sectionType {
        case PTAU_SECTION_HEADER: headerData = sectionBytes
        case PTAU_SECTION_G1_TAU: g1Data = sectionBytes
        case PTAU_SECTION_G2_TAU: g2Data = sectionBytes
        default: break
        }
        offset = sectionEnd
    }

    guard let header = headerData, header.count >= 12,
          let g1Raw = g1Data, let g2Raw = g2Data else { return nil }

    // Parse header: curve_id (4), power (4), ceremony_power (4)
    let curveId = readUInt32LE(header, offset: 0)
    let power = readUInt32LE(header, offset: 4)
    let degree = 1 << Int(power)  // 2^power G1 points (though file may have degree+1)

    let curve: CeremonyKZGCurve
    let g1PointSize: Int
    let g2PointSize: Int

    switch curveId {
    case PTAU_CURVE_BN254:
        curve = .bn254
        g1PointSize = 64   // 2 * 32 bytes uncompressed affine
        g2PointSize = 128  // 2 * 64 bytes uncompressed affine (Fp2)
    case PTAU_CURVE_BLS12381:
        curve = .bls12381
        g1PointSize = 96   // 2 * 48 bytes uncompressed affine
        g2PointSize = 192  // 2 * 96 bytes uncompressed affine (Fp2)
    default:
        return nil
    }

    let g1Count = g1Raw.count / g1PointSize
    let g2Count = g2Raw.count / g2PointSize
    guard g1Count > 0, g2Count >= 2 else { return nil }

    // The raw data is already in our internal format (uncompressed affine, big-endian coordinates)
    return StructuredReferenceString(curve: curve, g1Powers: Array(g1Raw.prefix(g1Count * g1PointSize)),
                                      g2Powers: Array(g2Raw.prefix(g2Count * g2PointSize)),
                                      degree: g1Count, g2Count: g2Count)
}

private func savePtau(_ srs: StructuredReferenceString) -> [UInt8]? {
    let curveId: UInt32
    let g1PointSize: Int
    let g2PointSize: Int
    var power: UInt32 = 0
    var n = srs.degree
    while n > 1 { n >>= 1; power += 1 }

    switch srs.curve {
    case .bn254:
        curveId = PTAU_CURVE_BN254
        g1PointSize = 64
        g2PointSize = 128
    case .bls12381:
        curveId = PTAU_CURVE_BLS12381
        g1PointSize = 96
        g2PointSize = 192
    }

    var result = [UInt8]()

    // Magic + version + num sections (3: header, g1, g2)
    result.append(contentsOf: PTAU_MAGIC)
    result.append(contentsOf: writeUInt32LE(PTAU_VERSION))
    result.append(contentsOf: writeUInt32LE(3))

    // Section 1: Header
    let headerPayload = writeUInt32LE(curveId) + writeUInt32LE(power) + writeUInt32LE(power)
    result.append(contentsOf: writeUInt32LE(PTAU_SECTION_HEADER))
    result.append(contentsOf: writeUInt64LE(UInt64(headerPayload.count)))
    result.append(contentsOf: headerPayload)

    // Section 2: G1 tau powers
    result.append(contentsOf: writeUInt32LE(PTAU_SECTION_G1_TAU))
    result.append(contentsOf: writeUInt64LE(UInt64(srs.g1Powers.count)))
    result.append(contentsOf: srs.g1Powers)

    // Section 3: G2 tau powers
    result.append(contentsOf: writeUInt32LE(PTAU_SECTION_G2_TAU))
    result.append(contentsOf: writeUInt64LE(UInt64(srs.g2Powers.count)))
    result.append(contentsOf: srs.g2Powers)

    return result
}

// MARK: - Ethereum KZG Ceremony Format

// The Ethereum EIP-4844 trusted setup format:
//   Concatenated file:
//     - Header: 8 bytes (g1Count as uint32 LE, g2Count as uint32 LE)
//     - g1Count compressed G1 points (48 bytes each, BLS12-381 ZCash format)
//     - g2Count compressed G2 points (96 bytes each, BLS12-381 ZCash format)
//
// This is the format used by ethereum/c-kzg-4844 trusted_setup.txt (but binary).
// The actual Ethereum ceremony outputs a JSON, but we support the binary layout
// which is more efficient for loading.

private func loadEthereumKZG(_ data: [UInt8]) -> StructuredReferenceString? {
    guard data.count >= 8 else { return nil }

    let g1Count = Int(readUInt32LE(data, offset: 0))
    let g2Count = Int(readUInt32LE(data, offset: 4))

    let g1CompressedSize = 48
    let g2CompressedSize = 96
    let expectedSize = 8 + g1Count * g1CompressedSize + g2Count * g2CompressedSize
    guard data.count >= expectedSize else { return nil }
    guard g1Count > 0, g2Count >= 2 else { return nil }

    // Decompress G1 points into uncompressed affine format (96 bytes each: x + y)
    var g1Bytes = [UInt8]()
    g1Bytes.reserveCapacity(g1Count * 96)
    var offset = 8
    for _ in 0..<g1Count {
        let compressed = Array(data[offset..<offset + g1CompressedSize])
        offset += g1CompressedSize
        guard let proj = bls12381G1Decompress(compressed),
              let aff = g1_381ToAffine(proj) else {
            return nil
        }
        g1Bytes.append(contentsOf: bls12381FpToBigEndian(aff.x))
        g1Bytes.append(contentsOf: bls12381FpToBigEndian(aff.y))
    }

    // Decompress G2 points into uncompressed affine format (192 bytes each)
    var g2Bytes = [UInt8]()
    g2Bytes.reserveCapacity(g2Count * 192)
    for _ in 0..<g2Count {
        let compressed = Array(data[offset..<offset + g2CompressedSize])
        offset += g2CompressedSize
        guard let proj = bls12381G2Decompress(compressed),
              let aff = g2_381ToAffine(proj) else {
            return nil
        }
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(aff.x.c0))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(aff.x.c1))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(aff.y.c0))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(aff.y.c1))
    }

    return StructuredReferenceString(curve: .bls12381, g1Powers: g1Bytes, g2Powers: g2Bytes,
                                      degree: g1Count, g2Count: g2Count)
}

private func saveEthereumKZG(_ srs: StructuredReferenceString) -> [UInt8]? {
    guard srs.curve == .bls12381 else { return nil }  // Ethereum KZG is BLS12-381 only

    guard let g1Points = srs.bls12381G1Points(),
          let g2Points = srs.bls12381G2Points() else { return nil }

    var result = [UInt8]()
    let g1CompressedSize = 48
    let g2CompressedSize = 96
    result.reserveCapacity(8 + g1Points.count * g1CompressedSize + g2Points.count * g2CompressedSize)

    // Header
    result.append(contentsOf: writeUInt32LE(UInt32(g1Points.count)))
    result.append(contentsOf: writeUInt32LE(UInt32(g2Points.count)))

    // Compressed G1 points
    for pt in g1Points {
        let proj = g1_381FromAffine(pt)
        result.append(contentsOf: bls12381G1Compress(proj))
    }

    // Compressed G2 points
    for pt in g2Points {
        let proj = g2_381FromAffine(pt)
        result.append(contentsOf: bls12381G2Compress(proj))
    }

    return result
}

// MARK: - File I/O Convenience

/// Load an SRS from a file path.
///
/// - Parameters:
///   - path: File system path to the SRS file.
///   - format: File format to parse.
/// - Returns: The parsed SRS, or nil on error.
public func loadSRSFromFile(path: String, format: SRSFileFormat) -> StructuredReferenceString? {
    guard let data = try? [UInt8](Data(contentsOf: URL(fileURLWithPath: path))) else {
        return nil
    }
    return loadSRS(from: data, format: format)
}

/// Save an SRS to a file path.
///
/// - Parameters:
///   - srs: The SRS to serialize.
///   - path: File system path to write.
///   - format: File format to write.
/// - Returns: true on success.
@discardableResult
public func saveSRSToFile(_ srs: StructuredReferenceString, path: String, format: SRSFileFormat) -> Bool {
    guard let data = saveSRS(srs, format: format) else { return false }
    do {
        try Data(data).write(to: URL(fileURLWithPath: path))
        return true
    } catch {
        return false
    }
}

// MARK: - Binary Helpers

private func readUInt32LE(_ data: [UInt8], offset: Int) -> UInt32 {
    UInt32(data[offset]) |
    (UInt32(data[offset + 1]) << 8) |
    (UInt32(data[offset + 2]) << 16) |
    (UInt32(data[offset + 3]) << 24)
}

private func readUInt64LE(_ data: [UInt8], offset: Int) -> UInt64 {
    UInt64(readUInt32LE(data, offset: offset)) |
    (UInt64(readUInt32LE(data, offset: offset + 4)) << 32)
}

private func writeUInt32LE(_ val: UInt32) -> [UInt8] {
    [UInt8(val & 0xFF), UInt8((val >> 8) & 0xFF),
     UInt8((val >> 16) & 0xFF), UInt8((val >> 24) & 0xFF)]
}

private func writeUInt64LE(_ val: UInt64) -> [UInt8] {
    writeUInt32LE(UInt32(val & 0xFFFFFFFF)) + writeUInt32LE(UInt32(val >> 32))
}
