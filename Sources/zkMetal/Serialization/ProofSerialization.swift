// Proof Serialization — Lightweight binary format for zkMetal proofs
// Binary writer/reader with format labels for interoperability.
// Fr elements are serialized as 32 raw bytes (8 UInt32 limbs, little-endian).
// Montgomery form is preserved — no conversion on serialize/deserialize.

import Foundation

// MARK: - Errors

public enum ProofSerializationError: Error, CustomStringConvertible {
    case truncatedData(expected: Int, available: Int)
    case wrongLabel(expected: String, got: String)
    case invalidUTF8
    case invalidHex
    case invalidBase64

    public var description: String {
        switch self {
        case .truncatedData(let expected, let available):
            return "Truncated data: need \(expected) bytes, have \(available)"
        case .wrongLabel(let expected, let got):
            return "Wrong format label: expected '\(expected)', got '\(got)'"
        case .invalidUTF8:
            return "Invalid UTF-8 in string label"
        case .invalidHex:
            return "Invalid hex string"
        case .invalidBase64:
            return "Invalid base64 string"
        }
    }
}

// MARK: - ProofWriter

/// Binary proof writer — appends field elements, integers, and raw bytes.
public class ProofWriter {
    private var data: [UInt8] = []

    public init() {}

    /// Write a field element (32 bytes: 8 UInt32 limbs in little-endian byte order)
    public func writeFr(_ value: Fr) {
        let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                     value.v.4, value.v.5, value.v.6, value.v.7]
        for limb in limbs {
            data.append(UInt8(limb & 0xFF))
            data.append(UInt8((limb >> 8) & 0xFF))
            data.append(UInt8((limb >> 16) & 0xFF))
            data.append(UInt8((limb >> 24) & 0xFF))
        }
    }

    /// Write multiple field elements (length-prefixed)
    public func writeFrArray(_ values: [Fr]) {
        writeUInt32(UInt32(values.count))
        for v in values { writeFr(v) }
    }

    /// Write an Fp element (32 bytes: 8 UInt32 limbs in little-endian byte order)
    public func writeFp(_ value: Fp) {
        let limbs = [value.v.0, value.v.1, value.v.2, value.v.3,
                     value.v.4, value.v.5, value.v.6, value.v.7]
        for limb in limbs {
            data.append(UInt8(limb & 0xFF))
            data.append(UInt8((limb >> 8) & 0xFF))
            data.append(UInt8((limb >> 16) & 0xFF))
            data.append(UInt8((limb >> 24) & 0xFF))
        }
    }

    /// Write a projective point (3 Fp elements: x, y, z)
    public func writePointProjective(_ p: PointProjective) {
        writeFp(p.x)
        writeFp(p.y)
        writeFp(p.z)
    }

    /// Write multiple projective points (length-prefixed)
    public func writePointProjectiveArray(_ points: [PointProjective]) {
        writeUInt32(UInt32(points.count))
        for p in points { writePointProjective(p) }
    }

    /// Write a sumcheck round tuple (3 Fr elements)
    public func writeSumcheckRound(_ round: (Fr, Fr, Fr)) {
        writeFr(round.0)
        writeFr(round.1)
        writeFr(round.2)
    }

    /// Write an array of sumcheck rounds (length-prefixed)
    public func writeSumcheckRounds(_ rounds: [(Fr, Fr, Fr)]) {
        writeUInt32(UInt32(rounds.count))
        for r in rounds { writeSumcheckRound(r) }
    }

    /// Write a raw UInt32 (4 bytes, little-endian)
    public func writeUInt32(_ value: UInt32) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 24) & 0xFF))
    }

    /// Write a raw UInt64 (8 bytes, little-endian)
    public func writeUInt64(_ value: UInt64) {
        for i in 0..<8 {
            data.append(UInt8((value >> (i * 8)) & 0xFF))
        }
    }

    /// Write a raw Int as UInt64
    public func writeInt(_ value: Int) {
        writeUInt64(UInt64(bitPattern: Int64(value)))
    }

    /// Write length-prefixed raw bytes
    public func writeBytes(_ bytes: [UInt8]) {
        writeUInt32(UInt32(bytes.count))
        data.append(contentsOf: bytes)
    }

    /// Write a string label (length-prefixed UTF-8)
    public func writeLabel(_ label: String) {
        let utf8 = Array(label.utf8)
        writeUInt32(UInt32(utf8.count))
        data.append(contentsOf: utf8)
    }

    /// Write an array of Int values (length-prefixed)
    public func writeIntArray(_ values: [Int]) {
        writeUInt32(UInt32(values.count))
        for v in values { writeInt(v) }
    }

    /// Get the serialized bytes
    public func finalize() -> [UInt8] { data }

    /// Current size in bytes
    public var size: Int { data.count }
}

// MARK: - ProofReader

/// Binary proof reader — reads field elements, integers, and raw bytes.
public class ProofReader {
    private let data: [UInt8]
    private var offset: Int = 0

    public init(_ data: [UInt8]) {
        self.data = data
    }

    /// Ensure at least `count` bytes remain
    private func ensureAvailable(_ count: Int) throws {
        if offset + count > data.count {
            throw ProofSerializationError.truncatedData(
                expected: count, available: data.count - offset)
        }
    }

    /// Read a field element (32 bytes: 8 UInt32 limbs in little-endian byte order)
    public func readFr() throws -> Fr {
        try ensureAvailable(32)
        var limbs = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            let b0 = UInt32(data[offset])
            let b1 = UInt32(data[offset + 1])
            let b2 = UInt32(data[offset + 2])
            let b3 = UInt32(data[offset + 3])
            limbs[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            offset += 4
        }
        return Fr(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                      limbs[4], limbs[5], limbs[6], limbs[7]))
    }

    /// Read a length-prefixed array of Fr elements
    public func readFrArray() throws -> [Fr] {
        let count = Int(try readUInt32())
        var result = [Fr]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(try readFr())
        }
        return result
    }

    /// Read an Fp element (32 bytes)
    public func readFp() throws -> Fp {
        try ensureAvailable(32)
        var limbs = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            let b0 = UInt32(data[offset])
            let b1 = UInt32(data[offset + 1])
            let b2 = UInt32(data[offset + 2])
            let b3 = UInt32(data[offset + 3])
            limbs[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            offset += 4
        }
        return Fp(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                      limbs[4], limbs[5], limbs[6], limbs[7]))
    }

    /// Read a projective point (3 Fp elements)
    public func readPointProjective() throws -> PointProjective {
        let x = try readFp()
        let y = try readFp()
        let z = try readFp()
        return PointProjective(x: x, y: y, z: z)
    }

    /// Read a length-prefixed array of projective points
    public func readPointProjectiveArray() throws -> [PointProjective] {
        let count = Int(try readUInt32())
        var result = [PointProjective]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(try readPointProjective())
        }
        return result
    }

    /// Read a sumcheck round tuple (3 Fr elements)
    public func readSumcheckRound() throws -> (Fr, Fr, Fr) {
        let a = try readFr()
        let b = try readFr()
        let c = try readFr()
        return (a, b, c)
    }

    /// Read a length-prefixed array of sumcheck rounds
    public func readSumcheckRounds() throws -> [(Fr, Fr, Fr)] {
        let count = Int(try readUInt32())
        var result = [(Fr, Fr, Fr)]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(try readSumcheckRound())
        }
        return result
    }

    /// Read a UInt32 (4 bytes, little-endian)
    public func readUInt32() throws -> UInt32 {
        try ensureAvailable(4)
        let b0 = UInt32(data[offset])
        let b1 = UInt32(data[offset + 1])
        let b2 = UInt32(data[offset + 2])
        let b3 = UInt32(data[offset + 3])
        offset += 4
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read a UInt64 (8 bytes, little-endian)
    public func readUInt64() throws -> UInt64 {
        try ensureAvailable(8)
        var val: UInt64 = 0
        for i in 0..<8 {
            val |= UInt64(data[offset + i]) << (i * 8)
        }
        offset += 8
        return val
    }

    /// Read an Int (from UInt64)
    public func readInt() throws -> Int {
        Int(Int64(bitPattern: try readUInt64()))
    }

    /// Read length-prefixed raw bytes
    public func readBytes() throws -> [UInt8] {
        let count = Int(try readUInt32())
        try ensureAvailable(count)
        let result = Array(data[offset..<(offset + count)])
        offset += count
        return result
    }

    /// Read a string label (length-prefixed UTF-8)
    public func readLabel() throws -> String {
        let bytes = try readBytes()
        guard let str = String(bytes: bytes, encoding: .utf8) else {
            throw ProofSerializationError.invalidUTF8
        }
        return str
    }

    /// Read and verify a specific label
    public func expectLabel(_ expected: String) throws {
        let got = try readLabel()
        if got != expected {
            throw ProofSerializationError.wrongLabel(expected: expected, got: got)
        }
    }

    /// Read a length-prefixed array of Int values
    public func readIntArray() throws -> [Int] {
        let count = Int(try readUInt32())
        var result = [Int]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(try readInt())
        }
        return result
    }

    /// Whether all data has been consumed
    public var isAtEnd: Bool { offset >= data.count }

    /// Remaining bytes
    public var remaining: Int { max(0, data.count - offset) }
}

// MARK: - Hex / Base64 Encoding Helpers

extension Array where Element == UInt8 {
    /// Convert bytes to lowercase hex string
    public func toHex() -> String {
        map { String(format: "%02x", $0) }.joined()
    }

    /// Parse a hex string into bytes. Returns nil on invalid input.
    public static func fromHex(_ hex: String) -> [UInt8]? {
        let chars = hex.map { $0 }
        guard chars.count % 2 == 0 else { return nil }
        var result = [UInt8]()
        result.reserveCapacity(chars.count / 2)
        var i = 0
        while i < chars.count {
            guard let hi = hexVal(chars[i]),
                  let lo = hexVal(chars[i + 1]) else { return nil }
            result.append((hi << 4) | lo)
            i += 2
        }
        return result
    }

    /// Convert bytes to base64 string
    public func toBase64() -> String {
        Data(self).base64EncodedString()
    }

    /// Parse a base64 string into bytes. Returns nil on invalid input.
    public static func fromBase64(_ b64: String) -> [UInt8]? {
        guard let data = Data(base64Encoded: b64) else { return nil }
        return Array(data)
    }

    private static func hexVal(_ c: Character) -> UInt8? {
        switch c {
        case "0"..."9": return UInt8(c.asciiValue! - Character("0").asciiValue!)
        case "a"..."f": return UInt8(c.asciiValue! - Character("a").asciiValue! + 10)
        case "A"..."F": return UInt8(c.asciiValue! - Character("A").asciiValue! + 10)
        default: return nil
        }
    }
}

// MARK: - FRI Serialization

extension FRICommitment {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("FRI-COMMIT-v1")
        w.writeUInt32(UInt32(layers.count))
        for layer in layers {
            w.writeFrArray(layer)
        }
        w.writeFrArray(roots)
        w.writeFrArray(betas)
        w.writeFr(finalValue)
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> FRICommitment {
        let r = ProofReader(data)
        try r.expectLabel("FRI-COMMIT-v1")
        let numLayers = Int(try r.readUInt32())
        var layers = [[Fr]]()
        layers.reserveCapacity(numLayers)
        for _ in 0..<numLayers {
            layers.append(try r.readFrArray())
        }
        let roots = try r.readFrArray()
        let betas = try r.readFrArray()
        let finalValue = try r.readFr()
        return FRICommitment(layers: layers, roots: roots, betas: betas, finalValue: finalValue)
    }
}

extension FRIQueryProof {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("FRI-QUERY-v1")
        w.writeUInt32(initialIndex)
        // Layer evals: array of pairs
        w.writeUInt32(UInt32(layerEvals.count))
        for (a, b) in layerEvals {
            w.writeFr(a)
            w.writeFr(b)
        }
        // Merkle paths: [layer][sibling_index][Fr]
        w.writeUInt32(UInt32(merklePaths.count))
        for layerPath in merklePaths {
            w.writeUInt32(UInt32(layerPath.count))
            for siblings in layerPath {
                w.writeFrArray(siblings)
            }
        }
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> FRIQueryProof {
        let r = ProofReader(data)
        try r.expectLabel("FRI-QUERY-v1")
        let initialIndex = try r.readUInt32()
        let numEvals = Int(try r.readUInt32())
        var layerEvals = [(Fr, Fr)]()
        layerEvals.reserveCapacity(numEvals)
        for _ in 0..<numEvals {
            let a = try r.readFr()
            let b = try r.readFr()
            layerEvals.append((a, b))
        }
        let numPaths = Int(try r.readUInt32())
        var merklePaths = [[[Fr]]]()
        merklePaths.reserveCapacity(numPaths)
        for _ in 0..<numPaths {
            let numSiblings = Int(try r.readUInt32())
            var layerPath = [[Fr]]()
            layerPath.reserveCapacity(numSiblings)
            for _ in 0..<numSiblings {
                layerPath.append(try r.readFrArray())
            }
            merklePaths.append(layerPath)
        }
        return FRIQueryProof(initialIndex: initialIndex, layerEvals: layerEvals, merklePaths: merklePaths)
    }
}

// MARK: - KZG Serialization

extension KZGProof {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("KZG-v1")
        w.writeFr(evaluation)
        w.writePointProjective(witness)
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> KZGProof {
        let r = ProofReader(data)
        try r.expectLabel("KZG-v1")
        let evaluation = try r.readFr()
        let witness = try r.readPointProjective()
        return KZGProof(evaluation: evaluation, witness: witness)
    }
}

// MARK: - IPA Serialization

extension IPAProof {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("IPA-v1")
        w.writePointProjectiveArray(L)
        w.writePointProjectiveArray(R)
        w.writeFr(a)
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> IPAProof {
        let r = ProofReader(data)
        try r.expectLabel("IPA-v1")
        let L = try r.readPointProjectiveArray()
        let R = try r.readPointProjectiveArray()
        let a = try r.readFr()
        return IPAProof(L: L, R: R, a: a)
    }
}

// MARK: - Sumcheck Serialization (standalone proof struct)

/// Serializable sumcheck proof (wraps the engine's return type)
public struct SumcheckProof {
    public let rounds: [(Fr, Fr, Fr)]
    public let finalEval: Fr

    public init(rounds: [(Fr, Fr, Fr)], finalEval: Fr) {
        self.rounds = rounds
        self.finalEval = finalEval
    }

    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("SUMCHECK-v1")
        w.writeSumcheckRounds(rounds)
        w.writeFr(finalEval)
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> SumcheckProof {
        let r = ProofReader(data)
        try r.expectLabel("SUMCHECK-v1")
        let rounds = try r.readSumcheckRounds()
        let finalEval = try r.readFr()
        return SumcheckProof(rounds: rounds, finalEval: finalEval)
    }
}

// MARK: - Lookup Serialization

extension LookupProof {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("LOOKUP-v1")
        w.writeFrArray(multiplicities)
        w.writeFr(beta)
        w.writeSumcheckRounds(lookupSumcheckRounds)
        w.writeSumcheckRounds(tableSumcheckRounds)
        w.writeFr(claimedSum)
        w.writeFr(lookupFinalEval)
        w.writeFr(tableFinalEval)
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> LookupProof {
        let r = ProofReader(data)
        try r.expectLabel("LOOKUP-v1")
        let multiplicities = try r.readFrArray()
        let beta = try r.readFr()
        let lookupRounds = try r.readSumcheckRounds()
        let tableRounds = try r.readSumcheckRounds()
        let claimedSum = try r.readFr()
        let lookupFinalEval = try r.readFr()
        let tableFinalEval = try r.readFr()
        return LookupProof(
            multiplicities: multiplicities, beta: beta,
            lookupSumcheckRounds: lookupRounds,
            tableSumcheckRounds: tableRounds,
            claimedSum: claimedSum,
            lookupFinalEval: lookupFinalEval,
            tableFinalEval: tableFinalEval)
    }
}

// MARK: - Lasso Serialization

extension SubtableProof {
    public func serialize(writer w: ProofWriter) {
        w.writeInt(chunkIndex)
        w.writeFrArray(readCounts)
        w.writeFr(beta)
        w.writeSumcheckRounds(readSumcheckRounds)
        w.writeSumcheckRounds(tableSumcheckRounds)
        w.writeFr(claimedSum)
        w.writeFr(readFinalEval)
        w.writeFr(tableFinalEval)
    }

    public static func deserialize(reader r: ProofReader) throws -> SubtableProof {
        let chunkIndex = try r.readInt()
        let readCounts = try r.readFrArray()
        let beta = try r.readFr()
        let readRounds = try r.readSumcheckRounds()
        let tableRounds = try r.readSumcheckRounds()
        let claimedSum = try r.readFr()
        let readFinalEval = try r.readFr()
        let tableFinalEval = try r.readFr()
        return SubtableProof(
            chunkIndex: chunkIndex, readCounts: readCounts, beta: beta,
            readSumcheckRounds: readRounds,
            tableSumcheckRounds: tableRounds,
            claimedSum: claimedSum,
            readFinalEval: readFinalEval,
            tableFinalEval: tableFinalEval)
    }
}

extension LassoProof {
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("LASSO-v1")
        w.writeInt(numChunks)
        w.writeUInt32(UInt32(subtableProofs.count))
        for sp in subtableProofs {
            sp.serialize(writer: w)
        }
        // indices: [[Int]] — numChunks arrays of Int
        w.writeUInt32(UInt32(indices.count))
        for chunkIndices in indices {
            w.writeIntArray(chunkIndices)
        }
        return w.finalize()
    }

    public static func deserialize(_ data: [UInt8]) throws -> LassoProof {
        let r = ProofReader(data)
        try r.expectLabel("LASSO-v1")
        let numChunks = try r.readInt()
        let numProofs = Int(try r.readUInt32())
        var subtableProofs = [SubtableProof]()
        subtableProofs.reserveCapacity(numProofs)
        for _ in 0..<numProofs {
            subtableProofs.append(try SubtableProof.deserialize(reader: r))
        }
        let numIndexArrays = Int(try r.readUInt32())
        var indices = [[Int]]()
        indices.reserveCapacity(numIndexArrays)
        for _ in 0..<numIndexArrays {
            indices.append(try r.readIntArray())
        }
        return LassoProof(numChunks: numChunks, subtableProofs: subtableProofs, indices: indices)
    }
}
