// Erasure Coding — Higher-level Reed-Solomon erasure coding for data availability
//
// Provides block-level encoding/decoding/verification on top of ReedSolomonEngine.
// - ErasureEncoder: chunks data into blocks, RS-encodes each, produces parity shards
// - ErasureDecoder: reconstructs from any sufficient subset of shards
// - ErasureVerifier: checks individual shards against a Merkle root commitment
//
// Configurable blowup rates: 2x, 4x, 8x.

import Foundation

// MARK: - ErasureShard

/// A single shard in an erasure-coded dataset.
/// Contains the shard index, the field elements for this shard across all blocks,
/// and a Merkle proof for verification.
public struct ErasureShard {
    /// Global shard index in the codeword (0..<n).
    public let index: Int

    /// One field element per block at this shard index.
    public let elements: [Fr]

    public init(index: Int, elements: [Fr]) {
        self.index = index
        self.elements = elements
    }
}

/// BabyBear variant of ErasureShard.
public struct ErasureShardBb {
    public let index: Int
    public let elements: [Bb]

    public init(index: Int, elements: [Bb]) {
        self.index = index
        self.elements = elements
    }
}

// MARK: - ErasureConfig

/// Configuration for erasure coding.
public struct ErasureConfig {
    /// Number of data elements per block for RS encoding.
    public let blockSize: Int

    /// Blowup factor (2, 4, or 8). Total shards per block = blockSize * blowupFactor (rounded to power of 2).
    public let blowupFactor: Int

    /// Total number of shards per block after RS encoding.
    public var totalShards: Int {
        nextPow2RS(blockSize * blowupFactor)
    }

    /// Minimum number of shards needed to reconstruct one block.
    public var minShards: Int {
        blockSize
    }

    public init(blockSize: Int, blowupFactor: Int = 2) {
        precondition(blockSize > 0, "Block size must be positive")
        precondition([2, 4, 8].contains(blowupFactor), "Blowup factor must be 2, 4, or 8")
        self.blockSize = blockSize
        self.blowupFactor = blowupFactor
    }

    /// Common configurations.
    public static let small = ErasureConfig(blockSize: 16, blowupFactor: 2)
    public static let medium = ErasureConfig(blockSize: 256, blowupFactor: 4)
    public static let large = ErasureConfig(blockSize: 4096, blowupFactor: 4)
}

// MARK: - ErasureEncoder (BN254 Fr)

/// Chunks data into blocks, RS-encodes each block, and produces shards.
/// Each shard contains one evaluation point from every block.
public class ErasureEncoder {
    public let rsEngine: ReedSolomonEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonEngine()
        self.config = config
    }

    /// Encode data into erasure-coded shards.
    ///
    /// - Parameter data: Input data as Fr field elements. Padded to a multiple of blockSize.
    /// - Returns: Array of ErasureShards (one per codeword position), and the Merkle root commitment.
    public func encode(data: [Fr]) throws -> (shards: [ErasureShard], merkleRoot: [UInt8]) {
        let blocks = chunkIntoBlocks(data)
        let numBlocks = blocks.count
        let n = config.totalShards

        // RS-encode each block
        var encodedBlocks = [[Fr]]()
        encodedBlocks.reserveCapacity(numBlocks)
        for block in blocks {
            let codeword = try rsEngine.encode(data: block, codeRate: config.blowupFactor)
            encodedBlocks.append(codeword)
        }

        // Transpose: create shards (one per codeword position)
        var shards = [ErasureShard]()
        shards.reserveCapacity(n)
        for i in 0..<n {
            var elements = [Fr]()
            elements.reserveCapacity(numBlocks)
            for block in encodedBlocks {
                elements.append(block[i])
            }
            shards.append(ErasureShard(index: i, elements: elements))
        }

        // Compute Merkle root over shard hashes for commitment
        let merkleRoot = computeMerkleRoot(shards: shards)

        return (shards: shards, merkleRoot: merkleRoot)
    }

    /// Split data into blocks of config.blockSize, padding the last block with zeros.
    private func chunkIntoBlocks(_ data: [Fr]) -> [[Fr]] {
        let bs = config.blockSize
        let numBlocks = max(1, (data.count + bs - 1) / bs)
        var blocks = [[Fr]]()
        blocks.reserveCapacity(numBlocks)

        for b in 0..<numBlocks {
            var block = [Fr](repeating: .zero, count: bs)
            for i in 0..<bs {
                let idx = b * bs + i
                if idx < data.count {
                    block[i] = data[idx]
                }
            }
            blocks.append(block)
        }
        return blocks
    }

    /// Compute a simple Merkle root by hashing each shard's elements.
    /// Uses a basic hash-chaining approach (Poseidon2 when available, SHA256 fallback).
    private func computeMerkleRoot(shards: [ErasureShard]) -> [UInt8] {
        // Hash each shard to a leaf digest
        var leaves = [[UInt8]]()
        leaves.reserveCapacity(shards.count)

        for shard in shards {
            var hasher = SHA256Hasher()
            // Hash shard index
            var idx = UInt32(shard.index)
            withUnsafeBytes(of: &idx) { hasher.update($0) }
            // Hash each element
            for element in shard.elements {
                let limbs = element.to64()
                for limb in limbs {
                    var l = limb
                    withUnsafeBytes(of: &l) { hasher.update($0) }
                }
            }
            leaves.append(hasher.finalize())
        }

        // Pad leaves to power of 2
        let n = nextPow2RS(leaves.count)
        while leaves.count < n {
            leaves.append([UInt8](repeating: 0, count: 32))
        }

        // Build Merkle tree bottom-up
        var layer = leaves
        while layer.count > 1 {
            var nextLayer = [[UInt8]]()
            nextLayer.reserveCapacity(layer.count / 2)
            for i in stride(from: 0, to: layer.count, by: 2) {
                var hasher = SHA256Hasher()
                hasher.update(layer[i])
                hasher.update(layer[i + 1])
                nextLayer.append(hasher.finalize())
            }
            layer = nextLayer
        }

        return layer[0]
    }
}

// MARK: - ErasureDecoder (BN254 Fr)

/// Reconstructs original data from any sufficient subset of shards.
public class ErasureDecoder {
    public let rsEngine: ReedSolomonEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonEngine()
        self.config = config
    }

    /// Reconstruct original data from a subset of shards.
    ///
    /// - Parameters:
    ///   - shards: Available shards (must have at least config.minShards).
    ///   - originalDataLen: Original number of data elements (before padding).
    /// - Returns: Reconstructed data as Fr field elements.
    public func decode(shards: [ErasureShard], originalDataLen: Int) throws -> [Fr] {
        guard shards.count >= config.minShards else {
            throw RSEngineError.insufficientSymbols
        }

        // Determine number of blocks from first shard
        guard let first = shards.first else {
            throw RSEngineError.insufficientSymbols
        }
        let numBlocks = first.elements.count

        // Reconstruct each block independently
        var allData = [Fr]()
        allData.reserveCapacity(numBlocks * config.blockSize)

        for b in 0..<numBlocks {
            // Build (index, value) pairs for this block
            let pairs: [(Int, Fr)] = shards.map { shard in
                (shard.index, shard.elements[b])
            }

            let blockCoeffs = try rsEngine.decode(codeword: pairs, dataLen: config.blockSize)
            allData.append(contentsOf: blockCoeffs)
        }

        // Trim to original length
        return Array(allData.prefix(originalDataLen))
    }
}

// MARK: - ErasureVerifier (BN254 Fr)

/// Verifies individual shards against a Merkle root commitment.
/// Can also verify that a shard is consistent with the RS encoding.
public class ErasureVerifier {
    public let rsEngine: ReedSolomonEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonEngine()
        self.config = config
    }

    /// Verify that a shard's hash is consistent with the given Merkle root.
    /// Uses a Merkle proof (array of sibling hashes along the path from leaf to root).
    ///
    /// - Parameters:
    ///   - shard: The shard to verify.
    ///   - merkleRoot: The expected Merkle root.
    ///   - proof: Sibling hashes from leaf to root. proof[0] is the sibling of the leaf.
    /// - Returns: true if the proof is valid.
    public func verifyShard(shard: ErasureShard, merkleRoot: [UInt8], proof: [[UInt8]]) -> Bool {
        // Hash the shard to get the leaf
        var hasher = SHA256Hasher()
        var idx = UInt32(shard.index)
        withUnsafeBytes(of: &idx) { hasher.update($0) }
        for element in shard.elements {
            let limbs = element.to64()
            for limb in limbs {
                var l = limb
                withUnsafeBytes(of: &l) { hasher.update($0) }
            }
        }
        var current = hasher.finalize()

        // Walk up the Merkle tree
        var position = shard.index
        for sibling in proof {
            var h = SHA256Hasher()
            if position & 1 == 0 {
                // current is left child
                h.update(current)
                h.update(sibling)
            } else {
                // current is right child
                h.update(sibling)
                h.update(current)
            }
            current = h.finalize()
            position >>= 1
        }

        return current == merkleRoot
    }

    /// Verify that a complete codeword (all shards for one block) is a valid RS encoding.
    ///
    /// - Parameters:
    ///   - codeword: Full codeword for one block (all n evaluations).
    ///   - dataLen: Expected polynomial degree bound.
    /// - Returns: true if the codeword is valid.
    public func verifyCodeword(codeword: [Fr], dataLen: Int) throws -> Bool {
        return try rsEngine.verify(codeword: codeword, dataLen: dataLen)
    }
}

// MARK: - ErasureEncoder (BabyBear)

/// BabyBear variant of the erasure encoder.
public class ErasureEncoderBb {
    public let rsEngine: ReedSolomonBbEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonBbEngine()
        self.config = config
    }

    /// Encode BabyBear data into erasure-coded shards.
    public func encode(data: [Bb]) throws -> (shards: [ErasureShardBb], merkleRoot: [UInt8]) {
        let blocks = chunkIntoBlocks(data)
        let numBlocks = blocks.count
        let n = config.totalShards

        var encodedBlocks = [[Bb]]()
        encodedBlocks.reserveCapacity(numBlocks)
        for block in blocks {
            let codeword = try rsEngine.encode(data: block, codeRate: config.blowupFactor)
            encodedBlocks.append(codeword)
        }

        var shards = [ErasureShardBb]()
        shards.reserveCapacity(n)
        for i in 0..<n {
            var elements = [Bb]()
            elements.reserveCapacity(numBlocks)
            for block in encodedBlocks {
                elements.append(block[i])
            }
            shards.append(ErasureShardBb(index: i, elements: elements))
        }

        let merkleRoot = computeMerkleRootBb(shards: shards)
        return (shards: shards, merkleRoot: merkleRoot)
    }

    private func chunkIntoBlocks(_ data: [Bb]) -> [[Bb]] {
        let bs = config.blockSize
        let numBlocks = max(1, (data.count + bs - 1) / bs)
        var blocks = [[Bb]]()
        blocks.reserveCapacity(numBlocks)

        for b in 0..<numBlocks {
            var block = [Bb](repeating: .zero, count: bs)
            for i in 0..<bs {
                let idx = b * bs + i
                if idx < data.count {
                    block[i] = data[idx]
                }
            }
            blocks.append(block)
        }
        return blocks
    }

    private func computeMerkleRootBb(shards: [ErasureShardBb]) -> [UInt8] {
        var leaves = [[UInt8]]()
        leaves.reserveCapacity(shards.count)

        for shard in shards {
            var hasher = SHA256Hasher()
            var idx = UInt32(shard.index)
            withUnsafeBytes(of: &idx) { hasher.update($0) }
            for element in shard.elements {
                var v = element.v
                withUnsafeBytes(of: &v) { hasher.update($0) }
            }
            leaves.append(hasher.finalize())
        }

        let n = nextPow2RS(leaves.count)
        while leaves.count < n {
            leaves.append([UInt8](repeating: 0, count: 32))
        }

        var layer = leaves
        while layer.count > 1 {
            var nextLayer = [[UInt8]]()
            nextLayer.reserveCapacity(layer.count / 2)
            for i in stride(from: 0, to: layer.count, by: 2) {
                var hasher = SHA256Hasher()
                hasher.update(layer[i])
                hasher.update(layer[i + 1])
                nextLayer.append(hasher.finalize())
            }
            layer = nextLayer
        }
        return layer[0]
    }
}

// MARK: - ErasureDecoder (BabyBear)

/// BabyBear variant of the erasure decoder.
public class ErasureDecoderBb {
    public let rsEngine: ReedSolomonBbEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonBbEngine()
        self.config = config
    }

    /// Reconstruct original BabyBear data from a subset of shards.
    public func decode(shards: [ErasureShardBb], originalDataLen: Int) throws -> [Bb] {
        guard shards.count >= config.minShards else {
            throw RSEngineError.insufficientSymbols
        }

        guard let first = shards.first else {
            throw RSEngineError.insufficientSymbols
        }
        let numBlocks = first.elements.count

        var allData = [Bb]()
        allData.reserveCapacity(numBlocks * config.blockSize)

        for b in 0..<numBlocks {
            let pairs: [(Int, Bb)] = shards.map { shard in
                (shard.index, shard.elements[b])
            }
            let blockCoeffs = try rsEngine.decode(codeword: pairs, dataLen: config.blockSize)
            allData.append(contentsOf: blockCoeffs)
        }

        return Array(allData.prefix(originalDataLen))
    }
}

// MARK: - ErasureVerifier (BabyBear)

/// BabyBear variant of the erasure verifier.
public class ErasureVerifierBb {
    public let rsEngine: ReedSolomonBbEngine
    public let config: ErasureConfig

    public init(config: ErasureConfig = .small) throws {
        self.rsEngine = try ReedSolomonBbEngine()
        self.config = config
    }

    /// Verify a BabyBear codeword is a valid RS encoding.
    public func verifyCodeword(codeword: [Bb], dataLen: Int) throws -> Bool {
        return try rsEngine.verify(codeword: codeword, dataLen: dataLen)
    }
}

// MARK: - Simple SHA256 Hasher (for Merkle commitments)

/// Lightweight SHA256 wrapper using CommonCrypto.
struct SHA256Hasher {
    private var data = [UInt8]()

    mutating func update(_ bytes: [UInt8]) {
        data.append(contentsOf: bytes)
    }

    mutating func update(_ buffer: UnsafeRawBufferPointer) {
        if let base = buffer.baseAddress {
            let ptr = base.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: buffer.count))
        }
    }

    func finalize() -> [UInt8] {
        // Use CC_SHA256 via the existing SHA256Engine or manual computation.
        // For portability, use a simple inline SHA256.
        return sha256Digest(data)
    }
}

/// Compute SHA256 digest of input bytes.
/// Uses CommonCrypto CC_SHA256 via bridging.
private func sha256Digest(_ input: [UInt8]) -> [UInt8] {
    var digest = [UInt8](repeating: 0, count: 32)
    input.withUnsafeBytes { inputPtr in
        digest.withUnsafeMutableBytes { digestPtr in
            // Use the system CC_SHA256
            _ = CC_SHA256_Wrapper(inputPtr.baseAddress!, Int32(input.count), digestPtr.baseAddress!)
        }
    }
    return digest
}

/// Wrapper around CommonCrypto SHA256.
/// We call through to the C function directly.
@_silgen_name("CC_SHA256")
private func CC_SHA256_Wrapper(_ data: UnsafeRawPointer!, _ len: Int32, _ md: UnsafeMutableRawPointer!) -> UnsafeMutableRawPointer!
