// SSZSerializer — Ethereum Simple Serialize (SSZ) support for proof data structures.
// Implements encoding/decoding per the Ethereum consensus spec (EIP-2537 compatible),
// enabling interoperability with Ethereum beacon chain clients (Prysm, Lighthouse, Teku, etc.).
//
// SSZ spec: https://ethereum.github.io/consensus-specs/ssz/simple-serialize
// Key properties:
//   - Fixed-size types are serialized inline (no length prefix)
//   - Variable-size types use 4-byte LE offset indirection
//   - Hash tree root uses SHA-256 binary Merkle tree over 32-byte chunks
//   - BLS12-381 points use ZCash compressed format (48-byte G1, 96-byte G2)

import Foundation

// MARK: - SSZ Errors

public enum SSZError: Error, CustomStringConvertible {
    case truncatedData(expected: Int, available: Int)
    case invalidLength(message: String)
    case invalidOffset(message: String)
    case invalidBoolValue(UInt8)
    case decodingFailed(String)

    public var description: String {
        switch self {
        case .truncatedData(let expected, let available):
            return "SSZ: truncated data, need \(expected) bytes, have \(available)"
        case .invalidLength(let msg):
            return "SSZ: invalid length — \(msg)"
        case .invalidOffset(let msg):
            return "SSZ: invalid offset — \(msg)"
        case .invalidBoolValue(let v):
            return "SSZ: invalid bool byte 0x\(String(v, radix: 16))"
        case .decodingFailed(let msg):
            return "SSZ: decoding failed — \(msg)"
        }
    }
}

// MARK: - SSZ Basic Type Encoding

/// SSZ encoder: writes bytes in SSZ wire format (little-endian, fixed-size).
public struct SSZWriter {
    private var data: [UInt8] = []

    public init() {}

    /// Current serialized size.
    public var size: Int { data.count }

    /// Finalize and return the encoded bytes.
    public func finalize() -> [UInt8] { data }

    // -- Fixed-size basic types --

    public mutating func writeBool(_ value: Bool) {
        data.append(value ? 1 : 0)
    }

    public mutating func writeUInt8(_ value: UInt8) {
        data.append(value)
    }

    public mutating func writeUInt16(_ value: UInt16) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
    }

    public mutating func writeUInt32(_ value: UInt32) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 24) & 0xFF))
    }

    public mutating func writeUInt64(_ value: UInt64) {
        for i in 0..<8 {
            data.append(UInt8((value >> (i * 8)) & 0xFF))
        }
    }

    /// Write a UInt128 as 16 bytes LE (stored as two UInt64 limbs: lo, hi).
    public mutating func writeUInt128(lo: UInt64, hi: UInt64) {
        writeUInt64(lo)
        writeUInt64(hi)
    }

    /// Write a UInt256 as 32 bytes LE (stored as four UInt64 limbs: l0..l3, little-endian).
    public mutating func writeUInt256(_ limbs: [UInt64]) {
        precondition(limbs.count == 4)
        for limb in limbs {
            writeUInt64(limb)
        }
    }

    // -- Fixed-size byte vectors --

    /// Write raw bytes (fixed-length, no length prefix in SSZ for fixed vectors).
    public mutating func writeFixedBytes(_ bytes: [UInt8]) {
        data.append(contentsOf: bytes)
    }

    /// Write a variable-length byte list (with 4-byte LE offset written separately by container logic).
    public mutating func writeBytes(_ bytes: [UInt8]) {
        data.append(contentsOf: bytes)
    }

    // -- SSZ offset helpers --

    /// Reserve space for a 4-byte offset and return the position where it was written.
    public mutating func reserveOffset() -> Int {
        let pos = data.count
        data.append(contentsOf: [0, 0, 0, 0])
        return pos
    }

    /// Patch a previously reserved offset to point at the current write position.
    public mutating func patchOffset(at position: Int) {
        let offset = UInt32(data.count)
        data[position]     = UInt8(offset & 0xFF)
        data[position + 1] = UInt8((offset >> 8) & 0xFF)
        data[position + 2] = UInt8((offset >> 16) & 0xFF)
        data[position + 3] = UInt8((offset >> 24) & 0xFF)
    }
}

/// SSZ decoder: reads bytes in SSZ wire format.
public struct SSZReader {
    private let data: [UInt8]
    private var offset: Int = 0

    public init(_ data: [UInt8]) {
        self.data = data
    }

    public var remaining: Int { max(0, data.count - offset) }
    public var isAtEnd: Bool { offset >= data.count }
    public var currentOffset: Int { offset }

    private mutating func ensureAvailable(_ count: Int) throws {
        if offset + count > data.count {
            throw SSZError.truncatedData(expected: count, available: data.count - offset)
        }
    }

    public mutating func readBool() throws -> Bool {
        try ensureAvailable(1)
        let v = data[offset]
        offset += 1
        switch v {
        case 0: return false
        case 1: return true
        default: throw SSZError.invalidBoolValue(v)
        }
    }

    public mutating func readUInt8() throws -> UInt8 {
        try ensureAvailable(1)
        let v = data[offset]
        offset += 1
        return v
    }

    public mutating func readUInt16() throws -> UInt16 {
        try ensureAvailable(2)
        let v = UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
        offset += 2
        return v
    }

    public mutating func readUInt32() throws -> UInt32 {
        try ensureAvailable(4)
        let v = UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) |
                (UInt32(data[offset + 2]) << 16) | (UInt32(data[offset + 3]) << 24)
        offset += 4
        return v
    }

    public mutating func readUInt64() throws -> UInt64 {
        try ensureAvailable(8)
        var v: UInt64 = 0
        for i in 0..<8 {
            v |= UInt64(data[offset + i]) << (i * 8)
        }
        offset += 8
        return v
    }

    public mutating func readUInt128() throws -> (lo: UInt64, hi: UInt64) {
        let lo = try readUInt64()
        let hi = try readUInt64()
        return (lo, hi)
    }

    public mutating func readUInt256() throws -> [UInt64] {
        var limbs = [UInt64]()
        limbs.reserveCapacity(4)
        for _ in 0..<4 {
            limbs.append(try readUInt64())
        }
        return limbs
    }

    /// Read a fixed number of bytes.
    public mutating func readFixedBytes(_ count: Int) throws -> [UInt8] {
        try ensureAvailable(count)
        let result = Array(data[offset..<(offset + count)])
        offset += count
        return result
    }

    /// Read a 4-byte LE offset value.
    public mutating func readOffset() throws -> UInt32 {
        try readUInt32()
    }

    /// Read all remaining bytes (for the last variable-length field in a container).
    public mutating func readRemainingBytes() -> [UInt8] {
        let result = Array(data[offset...])
        offset = data.count
        return result
    }

    /// Create a sub-reader for a slice of the data.
    public func subReader(from start: Int, count: Int) throws -> SSZReader {
        guard start + count <= data.count else {
            throw SSZError.truncatedData(expected: count, available: data.count - start)
        }
        return SSZReader(Array(data[start..<(start + count)]))
    }
}

// MARK: - SSZ BLS12-381 Point Containers

/// SSZ container for a compressed BLS12-381 G1 point (48 bytes fixed).
/// Matches the Bytes48 type used in Ethereum consensus spec for BLS pubkeys/signatures.
public struct SSZG1Point {
    /// 48-byte ZCash compressed G1 point.
    public let compressed: [UInt8]

    public init(compressed: [UInt8]) {
        precondition(compressed.count == 48)
        self.compressed = compressed
    }

    /// Create from a BLS12-381 G1 projective point.
    public init(point: G1Projective381) {
        self.compressed = bls12381G1Compress(point)
    }

    /// Decompress to a BLS12-381 G1 projective point.
    public func decompress() -> G1Projective381? {
        bls12381G1Decompress(compressed)
    }

    /// SSZ fixed size: 48 bytes.
    public static let sszFixedSize = 48

    /// SSZ encode (fixed-size: just the 48 bytes).
    public func sszEncode() -> [UInt8] {
        compressed
    }

    /// SSZ decode from 48 bytes.
    public static func sszDecode(_ data: [UInt8]) throws -> SSZG1Point {
        guard data.count == 48 else {
            throw SSZError.invalidLength(message: "G1 point must be 48 bytes, got \(data.count)")
        }
        return SSZG1Point(compressed: data)
    }
}

/// SSZ container for a compressed BLS12-381 G2 point (96 bytes fixed).
/// Matches the Bytes96 type used in Ethereum consensus spec for BLS signatures.
public struct SSZG2Point {
    /// 96-byte ZCash compressed G2 point.
    public let compressed: [UInt8]

    public init(compressed: [UInt8]) {
        precondition(compressed.count == 96)
        self.compressed = compressed
    }

    /// Create from a BLS12-381 G2 projective point.
    public init(point: G2Projective381) {
        self.compressed = bls12381G2Compress(point)
    }

    /// Decompress to a BLS12-381 G2 projective point.
    public func decompress() -> G2Projective381? {
        bls12381G2Decompress(compressed)
    }

    /// SSZ fixed size: 96 bytes.
    public static let sszFixedSize = 96

    /// SSZ encode (fixed-size: just the 96 bytes).
    public func sszEncode() -> [UInt8] {
        compressed
    }

    /// SSZ decode from 96 bytes.
    public static func sszDecode(_ data: [UInt8]) throws -> SSZG2Point {
        guard data.count == 96 else {
            throw SSZError.invalidLength(message: "G2 point must be 96 bytes, got \(data.count)")
        }
        return SSZG2Point(compressed: data)
    }
}

// MARK: - SSZ Groth16 Proof Container (circom/snarkjs compatible)

/// SSZ-encoded Groth16 proof matching the circom/snarkjs format.
/// This uses BN254 curve points (the standard for on-chain Groth16 verification).
///
/// SSZ layout (all fixed-size, total 256 bytes):
///   pi_a:  64 bytes (2 x 32-byte Fp elements, x and y, big-endian)
///   pi_b: 128 bytes (2 x 2 x 32-byte Fp2 elements for G2 point, x and y)
///   pi_c:  64 bytes (2 x 32-byte Fp elements, x and y, big-endian)
public struct SSZGroth16Proof {
    /// G1 point A: 2 x 32 bytes (affine x, y as big-endian field elements).
    public let piA: [UInt8]  // 64 bytes
    /// G2 point B: 4 x 32 bytes (affine x = (c0, c1), y = (c0, c1), big-endian).
    public let piB: [UInt8]  // 128 bytes
    /// G1 point C: 2 x 32 bytes (affine x, y as big-endian field elements).
    public let piC: [UInt8]  // 64 bytes

    public static let sszFixedSize = 256

    public init(piA: [UInt8], piB: [UInt8], piC: [UInt8]) {
        precondition(piA.count == 64)
        precondition(piB.count == 128)
        precondition(piC.count == 64)
        self.piA = piA
        self.piB = piB
        self.piC = piC
    }

    /// Create from a zkMetal Groth16Proof (BN254).
    public init(proof: Groth16Proof) {
        // Convert projective to affine and serialize as big-endian
        self.piA = SSZGroth16Proof.serializeG1(proof.a)
        self.piB = SSZGroth16Proof.serializeG2(proof.b)
        self.piC = SSZGroth16Proof.serializeG1(proof.c)
    }

    /// Decode back to a zkMetal Groth16Proof.
    public func toGroth16Proof() -> Groth16Proof? {
        guard let a = SSZGroth16Proof.deserializeG1(piA),
              let b = SSZGroth16Proof.deserializeG2(piB),
              let c = SSZGroth16Proof.deserializeG1(piC) else { return nil }
        return Groth16Proof(a: a, b: b, c: c)
    }

    /// SSZ encode: concatenate fixed fields.
    public func sszEncode() -> [UInt8] {
        piA + piB + piC
    }

    /// SSZ decode from 256 bytes.
    public static func sszDecode(_ data: [UInt8]) throws -> SSZGroth16Proof {
        guard data.count == sszFixedSize else {
            throw SSZError.invalidLength(message: "Groth16 proof must be \(sszFixedSize) bytes, got \(data.count)")
        }
        let piA = Array(data[0..<64])
        let piB = Array(data[64..<192])
        let piC = Array(data[192..<256])
        return SSZGroth16Proof(piA: piA, piB: piB, piC: piC)
    }

    // -- Internal helpers --

    /// Serialize a BN254 G1 projective point to 64 bytes (x || y, each 32-byte big-endian).
    private static func serializeG1(_ p: PointProjective) -> [UInt8] {
        guard let aff = pointToAffine(p) else {
            return [UInt8](repeating: 0, count: 64)
        }
        let xLimbs = fpToInt(aff.x)
        let yLimbs = fpToInt(aff.y)
        return limbs4ToBE32(xLimbs) + limbs4ToBE32(yLimbs)
    }

    /// Deserialize a BN254 G1 point from 64 bytes (x || y, big-endian).
    private static func deserializeG1(_ data: [UInt8]) -> PointProjective? {
        guard data.count == 64 else { return nil }
        // Check for identity (all zeros)
        if data.allSatisfy({ $0 == 0 }) { return pointIdentity() }
        let xLimbs = be32ToLimbs4(Array(data[0..<32]))
        let yLimbs = be32ToLimbs4(Array(data[32..<64]))
        let x = fpMul(Fp.from64(xLimbs), Fp.from64(Fp.R2_MOD_P))
        let y = fpMul(Fp.from64(yLimbs), Fp.from64(Fp.R2_MOD_P))
        return PointProjective(x: x, y: y, z: .one)
    }

    /// Serialize a BN254 G2 projective point to 128 bytes
    /// (x.c0 || x.c1 || y.c0 || y.c1, each 32-byte big-endian).
    private static func serializeG2(_ p: G2ProjectivePoint) -> [UInt8] {
        guard let aff = g2ToAffine(p) else {
            return [UInt8](repeating: 0, count: 128)
        }
        let xc0 = fpToInt(aff.x.c0)
        let xc1 = fpToInt(aff.x.c1)
        let yc0 = fpToInt(aff.y.c0)
        let yc1 = fpToInt(aff.y.c1)
        return limbs4ToBE32(xc0) + limbs4ToBE32(xc1) + limbs4ToBE32(yc0) + limbs4ToBE32(yc1)
    }

    /// Deserialize a BN254 G2 point from 128 bytes.
    private static func deserializeG2(_ data: [UInt8]) -> G2ProjectivePoint? {
        guard data.count == 128 else { return nil }
        if data.allSatisfy({ $0 == 0 }) { return g2Identity() }
        let xc0Limbs = be32ToLimbs4(Array(data[0..<32]))
        let xc1Limbs = be32ToLimbs4(Array(data[32..<64]))
        let yc0Limbs = be32ToLimbs4(Array(data[64..<96]))
        let yc1Limbs = be32ToLimbs4(Array(data[96..<128]))
        let xc0 = fpMul(Fp.from64(xc0Limbs), Fp.from64(Fp.R2_MOD_P))
        let xc1 = fpMul(Fp.from64(xc1Limbs), Fp.from64(Fp.R2_MOD_P))
        let yc0 = fpMul(Fp.from64(yc0Limbs), Fp.from64(Fp.R2_MOD_P))
        let yc1 = fpMul(Fp.from64(yc1Limbs), Fp.from64(Fp.R2_MOD_P))
        return G2ProjectivePoint(
            x: Fp2(c0: xc0, c1: xc1),
            y: Fp2(c0: yc0, c1: yc1),
            z: .one
        )
    }
}

// MARK: - SSZ Hash Tree Root (SHA-256 Merkleization per Ethereum spec)

/// Compute the SSZ hash tree root of serialized data.
/// Per the Ethereum SSZ spec:
///   1. Split data into 32-byte chunks (zero-pad the last chunk if needed)
///   2. Build a binary Merkle tree using SHA-256
///   3. If the number of chunks < next power of 2, pad with zero-hash chunks
///   4. Return the root
///
/// Uses the CPU SHA-256 implementation (no GPU needed for Merkle proofs).
public enum SSZHashTreeRoot {

    /// Compute the hash tree root of raw serialized bytes.
    /// This is the core SSZ Merkleization function.
    public static func hashTreeRoot(data: [UInt8], limit: Int? = nil) -> [UInt8] {
        let chunks = chunkify(data)
        return merkleize(chunks, limit: limit)
    }

    /// Compute the hash tree root of a fixed-size SSZ basic value.
    /// The value is packed into a single 32-byte chunk (zero-padded).
    public static func hashTreeRootBasic(_ data: [UInt8]) -> [UInt8] {
        precondition(data.count <= 32)
        var chunk = [UInt8](repeating: 0, count: 32)
        for i in 0..<data.count { chunk[i] = data[i] }
        return chunk  // Single chunk = its own root
    }

    /// Compute the hash tree root of an SSZ container (list of field roots).
    /// Each field should already be its own hash tree root (32 bytes).
    public static func hashTreeRootContainer(fieldRoots: [[UInt8]]) -> [UInt8] {
        precondition(fieldRoots.allSatisfy { $0.count == 32 })
        return merkleize(fieldRoots, limit: nil)
    }

    /// Mix in a length value with a root (for SSZ lists).
    /// result = SHA-256(root || length_as_le_256bit)
    public static func mixInLength(root: [UInt8], length: UInt64) -> [UInt8] {
        var lengthChunk = [UInt8](repeating: 0, count: 32)
        for i in 0..<8 {
            lengthChunk[i] = UInt8((length >> (i * 8)) & 0xFF)
        }
        return sha256(root + lengthChunk)
    }

    // MARK: Chunking

    /// Split data into 32-byte chunks, zero-padding the last chunk.
    public static func chunkify(_ data: [UInt8]) -> [[UInt8]] {
        if data.isEmpty {
            return [[UInt8](repeating: 0, count: 32)]
        }
        let chunkCount = (data.count + 31) / 32
        var chunks = [[UInt8]]()
        chunks.reserveCapacity(chunkCount)
        for i in 0..<chunkCount {
            let start = i * 32
            let end = min(start + 32, data.count)
            var chunk = [UInt8](repeating: 0, count: 32)
            for j in start..<end {
                chunk[j - start] = data[j]
            }
            chunks.append(chunk)
        }
        return chunks
    }

    // MARK: Merkleization

    /// Zero hashes cache: zeroHashes[i] = Merkle root of 2^i zero leaves.
    private static let zeroHashes: [[UInt8]] = {
        var hashes = [[UInt8]]()
        hashes.append([UInt8](repeating: 0, count: 32))  // zeroHashes[0]
        for _ in 1..<64 {
            let prev = hashes.last!
            hashes.append(sha256(prev + prev))
        }
        return hashes
    }()

    /// Merkleize a list of 32-byte chunks into a single root.
    /// If limit is provided, the tree is padded to at least `limit` leaves.
    public static func merkleize(_ chunks: [[UInt8]], limit: Int?) -> [UInt8] {
        let effectiveLimit = limit ?? chunks.count
        let leafCount = max(effectiveLimit, 1)
        // Round up to next power of 2
        let treeLeaves = nextPowerOf2(leafCount)
        let depth = treeLeaves == 1 ? 0 : Int(log2(Double(treeLeaves)))

        if chunks.isEmpty {
            return zeroHashes[depth]
        }

        // Build bottom layer: chunks + zero padding
        var layer = [UInt8]()
        layer.reserveCapacity(treeLeaves * 32)
        for chunk in chunks {
            layer.append(contentsOf: chunk)
        }
        // Pad with zero chunks
        let zeroChunk = [UInt8](repeating: 0, count: 32)
        for _ in chunks.count..<treeLeaves {
            layer.append(contentsOf: zeroChunk)
        }

        // Build tree bottom-up
        var currentSize = treeLeaves
        while currentSize > 1 {
            var nextLayer = [UInt8]()
            let parentCount = currentSize / 2
            nextLayer.reserveCapacity(parentCount * 32)
            for i in 0..<parentCount {
                let left = Array(layer[(i * 2) * 32..<(i * 2 + 1) * 32])
                let right = Array(layer[(i * 2 + 1) * 32..<(i * 2 + 2) * 32])
                nextLayer.append(contentsOf: sha256(left + right))
            }
            layer = nextLayer
            currentSize = parentCount
        }

        return Array(layer[0..<32])
    }

    /// Next power of 2 >= n.
    internal static func nextPowerOf2(_ n: Int) -> Int {
        if n <= 1 { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        return v + 1
    }
}

// MARK: - SSZ Merkle Proof (Binary Tree Proof-of-Inclusion)

/// A Merkle proof-of-inclusion for an SSZ Merkleized tree.
/// Proves that a leaf at a given index is included in a tree with a known root.
public struct SSZMerkleProof {
    /// The leaf value being proven (32 bytes).
    public let leaf: [UInt8]
    /// The generalized index of the leaf in the tree (1-indexed, root = 1).
    public let generalizedIndex: UInt64
    /// Sibling hashes from leaf to root (bottom-up). Length = depth of tree.
    public let branch: [[UInt8]]

    public init(leaf: [UInt8], generalizedIndex: UInt64, branch: [[UInt8]]) {
        precondition(leaf.count == 32)
        precondition(branch.allSatisfy { $0.count == 32 })
        self.leaf = leaf
        self.generalizedIndex = generalizedIndex
        self.branch = branch
    }

    /// Verify this Merkle proof against a known root.
    public func verify(root: [UInt8]) -> Bool {
        guard root.count == 32 else { return false }
        var current = leaf
        var index = generalizedIndex
        for sibling in branch {
            if index & 1 == 0 {
                // Current node is a left child
                current = sha256(current + sibling)
            } else {
                // Current node is a right child
                current = sha256(sibling + current)
            }
            index >>= 1
        }
        return current == root
    }

    /// SSZ-encode the proof (for transmission).
    /// Format: leaf (32) || generalizedIndex (8 LE) || branchLength (4 LE) || branch (n * 32).
    public func sszEncode() -> [UInt8] {
        var out = SSZWriter()
        out.writeFixedBytes(leaf)
        out.writeUInt64(generalizedIndex)
        out.writeUInt32(UInt32(branch.count))
        for sibling in branch {
            out.writeFixedBytes(sibling)
        }
        return out.finalize()
    }

    /// SSZ-decode a Merkle proof.
    public static func sszDecode(_ data: [UInt8]) throws -> SSZMerkleProof {
        var reader = SSZReader(data)
        let leaf = try reader.readFixedBytes(32)
        let gIndex = try reader.readUInt64()
        let branchLen = Int(try reader.readUInt32())
        var branch = [[UInt8]]()
        branch.reserveCapacity(branchLen)
        for _ in 0..<branchLen {
            branch.append(try reader.readFixedBytes(32))
        }
        return SSZMerkleProof(leaf: leaf, generalizedIndex: gIndex, branch: branch)
    }
}

/// Generate SSZ Merkle proofs from a set of chunks.
public enum SSZMerkleProofGenerator {

    /// Build a full Merkle tree from chunks and generate a proof for the leaf at `leafIndex`.
    /// Returns nil if leafIndex is out of range.
    public static func generateProof(chunks: [[UInt8]], leafIndex: Int) -> SSZMerkleProof? {
        guard !chunks.isEmpty, leafIndex < chunks.count else { return nil }
        precondition(chunks.allSatisfy { $0.count == 32 })

        let treeLeaves = SSZHashTreeRoot.nextPowerOf2(chunks.count)
        let depth = treeLeaves == 1 ? 0 : Int(log2(Double(treeLeaves)))

        // Flat byte buffer for all tree nodes.
        // Layout: [level0: treeLeaves * 32] [level1: treeLeaves/2 * 32] ... [root: 32]
        // Total nodes = 2 * treeLeaves - 1
        let totalNodes = 2 * treeLeaves - 1
        var tree = [UInt8](repeating: 0, count: totalNodes * 32)

        // Copy leaves into level 0
        for i in 0..<chunks.count {
            let dst = i * 32
            for j in 0..<32 { tree[dst + j] = chunks[i][j] }
        }

        // Track where each level starts in the flat buffer
        var levelOffsets = [Int]()
        levelOffsets.reserveCapacity(depth + 1)
        var offset = 0
        var levelSize = treeLeaves
        for _ in 0...depth {
            levelOffsets.append(offset)
            offset += levelSize * 32
            levelSize = max(levelSize / 2, 1)
        }

        // Build internal levels
        levelSize = treeLeaves
        for level in 0..<depth {
            let parentCount = levelSize / 2
            let srcBase = levelOffsets[level]
            let dstBase = levelOffsets[level + 1]
            for i in 0..<parentCount {
                let leftStart = srcBase + (i * 2) * 32
                let rightStart = srcBase + (i * 2 + 1) * 32
                let left = Array(tree[leftStart..<leftStart + 32])
                let right = Array(tree[rightStart..<rightStart + 32])
                let hash = sha256(left + right)
                let dst = dstBase + i * 32
                for j in 0..<32 { tree[dst + j] = hash[j] }
            }
            levelSize = parentCount
        }

        // Extract branch (sibling at each level)
        var branch = [[UInt8]]()
        branch.reserveCapacity(depth)
        var idx = leafIndex
        for level in 0..<depth {
            let siblingIdx = idx ^ 1
            let siblingStart = levelOffsets[level] + siblingIdx * 32
            branch.append(Array(tree[siblingStart..<siblingStart + 32]))
            idx >>= 1
        }

        // Generalized index: leaf position in 1-indexed complete binary tree
        let generalizedIndex = UInt64(treeLeaves + leafIndex)

        let leafStart = leafIndex * 32
        let leafData = Array(tree[leafStart..<leafStart + 32])
        return SSZMerkleProof(leaf: leafData, generalizedIndex: generalizedIndex, branch: branch)
    }
}

// MARK: - SSZ Encoding for Existing Proof Types

public extension SSZG1Point {
    /// Compute the SSZ hash tree root for this G1 point.
    /// 48 bytes -> 2 chunks of 32 bytes (with zero padding).
    func hashTreeRoot() -> [UInt8] {
        SSZHashTreeRoot.hashTreeRoot(data: compressed)
    }
}

public extension SSZG2Point {
    /// Compute the SSZ hash tree root for this G2 point.
    /// 96 bytes -> 3 chunks of 32 bytes.
    func hashTreeRoot() -> [UInt8] {
        SSZHashTreeRoot.hashTreeRoot(data: compressed)
    }
}

public extension SSZGroth16Proof {
    /// Compute the SSZ hash tree root for a Groth16 proof.
    /// Container with 3 fields: piA, piB, piC.
    func hashTreeRoot() -> [UInt8] {
        let fieldRoots = [
            SSZHashTreeRoot.hashTreeRoot(data: piA),
            SSZHashTreeRoot.hashTreeRoot(data: piB),
            SSZHashTreeRoot.hashTreeRoot(data: piC),
        ]
        return SSZHashTreeRoot.hashTreeRootContainer(fieldRoots: fieldRoots)
    }
}

// MARK: - Convenience: Groth16Proof SSZ Extension

public extension Groth16Proof {
    /// Encode this proof in SSZ format (256 bytes).
    func sszEncode() -> [UInt8] {
        SSZGroth16Proof(proof: self).sszEncode()
    }

    /// Decode a Groth16 proof from SSZ format.
    static func sszDecode(_ data: [UInt8]) throws -> Groth16Proof {
        let ssz = try SSZGroth16Proof.sszDecode(data)
        guard let proof = ssz.toGroth16Proof() else {
            throw SSZError.decodingFailed("Failed to reconstruct Groth16 proof from SSZ data")
        }
        return proof
    }

    /// Compute the SSZ hash tree root for this proof.
    func sszHashTreeRoot() -> [UInt8] {
        SSZGroth16Proof(proof: self).hashTreeRoot()
    }
}

// MARK: - 32-byte Big-Endian Conversion Helpers (for 4-limb UInt64 fields)

/// Convert 4 x UInt64 limbs (little-endian) to 32-byte big-endian.
private func limbs4ToBE32(_ limbs: [UInt64]) -> [UInt8] {
    precondition(limbs.count == 4)
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let limb = limbs[3 - i]
        bytes[i * 8 + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[i * 8 + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[i * 8 + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[i * 8 + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[i * 8 + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[i * 8 + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[i * 8 + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[i * 8 + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Parse 32-byte big-endian into 4 x UInt64 limbs (little-endian).
private func be32ToLimbs4(_ bytes: [UInt8]) -> [UInt64] {
    precondition(bytes.count == 32)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let base = (3 - i) * 8
        limbs[i] = (UInt64(bytes[base]) << 56) | (UInt64(bytes[base + 1]) << 48) |
                   (UInt64(bytes[base + 2]) << 40) | (UInt64(bytes[base + 3]) << 32) |
                   (UInt64(bytes[base + 4]) << 24) | (UInt64(bytes[base + 5]) << 16) |
                   (UInt64(bytes[base + 6]) << 8) | UInt64(bytes[base + 7])
    }
    return limbs
}

// MARK: - SSZ Serializer Facade

/// High-level SSZ serialization interface for Ethereum consensus interoperability.
public enum SSZSerializer {

    // MARK: BLS12-381 Points

    /// Encode a BLS12-381 G1 point in SSZ format (48 bytes compressed).
    public static func encodeG1(_ point: G1Projective381) -> [UInt8] {
        SSZG1Point(point: point).sszEncode()
    }

    /// Decode a BLS12-381 G1 point from SSZ format.
    public static func decodeG1(_ data: [UInt8]) throws -> G1Projective381 {
        let ssz = try SSZG1Point.sszDecode(data)
        guard let point = ssz.decompress() else {
            throw SSZError.decodingFailed("Invalid compressed G1 point")
        }
        return point
    }

    /// Encode a BLS12-381 G2 point in SSZ format (96 bytes compressed).
    public static func encodeG2(_ point: G2Projective381) -> [UInt8] {
        SSZG2Point(point: point).sszEncode()
    }

    /// Decode a BLS12-381 G2 point from SSZ format.
    public static func decodeG2(_ data: [UInt8]) throws -> G2Projective381 {
        let ssz = try SSZG2Point.sszDecode(data)
        guard let point = ssz.decompress() else {
            throw SSZError.decodingFailed("Invalid compressed G2 point")
        }
        return point
    }

    // MARK: Groth16 Proof

    /// Encode a Groth16 proof in SSZ format (256 bytes).
    public static func encodeGroth16Proof(_ proof: Groth16Proof) -> [UInt8] {
        proof.sszEncode()
    }

    /// Decode a Groth16 proof from SSZ format.
    public static func decodeGroth16Proof(_ data: [UInt8]) throws -> Groth16Proof {
        try Groth16Proof.sszDecode(data)
    }

    // MARK: Hash Tree Root

    /// Compute the SSZ hash tree root of raw bytes.
    public static func hashTreeRoot(_ data: [UInt8]) -> [UInt8] {
        SSZHashTreeRoot.hashTreeRoot(data: data)
    }

    /// Compute the SSZ hash tree root for a Groth16 proof.
    public static func hashTreeRootGroth16(_ proof: Groth16Proof) -> [UInt8] {
        proof.sszHashTreeRoot()
    }

    /// Compute the SSZ hash tree root for a G1 point.
    public static func hashTreeRootG1(_ point: G1Projective381) -> [UInt8] {
        SSZG1Point(point: point).hashTreeRoot()
    }

    /// Compute the SSZ hash tree root for a G2 point.
    public static func hashTreeRootG2(_ point: G2Projective381) -> [UInt8] {
        SSZG2Point(point: point).hashTreeRoot()
    }

    // MARK: Merkle Proofs

    /// Generate an SSZ Merkle proof for a leaf at `leafIndex` within chunked data.
    public static func generateMerkleProof(data: [UInt8], leafIndex: Int) -> SSZMerkleProof? {
        let chunks = SSZHashTreeRoot.chunkify(data)
        return SSZMerkleProofGenerator.generateProof(chunks: chunks, leafIndex: leafIndex)
    }

    /// Verify an SSZ Merkle proof against a known root.
    public static func verifyMerkleProof(_ proof: SSZMerkleProof, root: [UInt8]) -> Bool {
        proof.verify(root: root)
    }

    // MARK: Basic Type Helpers

    /// SSZ-encode a UInt64 (8 bytes LE).
    public static func encodeUInt64(_ value: UInt64) -> [UInt8] {
        var w = SSZWriter()
        w.writeUInt64(value)
        return w.finalize()
    }

    /// SSZ-encode a UInt256 (32 bytes LE from 4 UInt64 limbs).
    public static func encodeUInt256(_ limbs: [UInt64]) -> [UInt8] {
        var w = SSZWriter()
        w.writeUInt256(limbs)
        return w.finalize()
    }

    /// SSZ-encode a boolean (1 byte).
    public static func encodeBool(_ value: Bool) -> [UInt8] {
        [value ? 1 : 0]
    }
}
