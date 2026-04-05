// Data Availability Sampling (DAS) Engine
// Demonstrates Ethereum Danksharding-style data availability sampling.
// Encodes blobs with RS redundancy, supports random sampling and reconstruction.
//
// Uses BabyBear NTT-based RS for efficient blob encoding:
// - Blob bytes -> field elements -> RS encode (NTT) -> extended data
// - Random column sampling to verify data availability
// - Full reconstruction from sufficient samples

import Foundation
import Metal

public class DataAvailabilitySampler {
    public let rsEngine: ReedSolomonNTTEngine
    public let expansionFactor: Int

    /// Initialize with an expansion factor (default 2x = 50% redundancy).
    public init(expansionFactor: Int = 2) throws {
        self.rsEngine = try ReedSolomonNTTEngine()
        self.expansionFactor = expansionFactor
    }

    /// Encode a blob (raw bytes) into extended data with RS redundancy.
    /// Each BabyBear element holds up to 30 bits of data.
    /// Returns (encoded shards, original element count).
    public func encodeBlob(data: [UInt8]) throws -> (shards: [Bb], originalK: Int, totalN: Int) {
        // Pack bytes into BabyBear field elements (3 bytes per element, safe under p)
        let elements = packBytesToBabyBear(data)
        let k = elements.count
        let n = nextPow2(k * expansionFactor)

        let shards = try rsEngine.encode(data: elements, expansionFactor: expansionFactor)
        return (shards: shards, originalK: k, totalN: n)
    }

    /// Sample random shard indices and return their values.
    /// In a real DAS system, these would be fetched from the network.
    public func sampleShards(allShards: [Bb], numSamples: Int) -> [(index: Int, value: Bb)] {
        let n = allShards.count
        var sampled = [(index: Int, value: Bb)]()
        var used = Set<Int>()
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern)

        while sampled.count < min(numSamples, n) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 33) % n
            if !used.contains(idx) {
                used.insert(idx)
                sampled.append((index: idx, value: allShards[idx]))
            }
        }
        return sampled
    }

    /// Verify that sampled shards are consistent with a known encoding.
    /// In production this would use KZG commitments; here we re-encode and check.
    public func verifySamples(samples: [(index: Int, value: Bb)],
                              allShards: [Bb]) -> Bool {
        for (idx, val) in samples {
            if idx >= allShards.count || allShards[idx].v != val.v {
                return false
            }
        }
        return true
    }

    /// Reconstruct original data from sufficient samples.
    /// Requires at least originalK samples.
    public func reconstruct(samples: [(index: Int, value: Bb)],
                           originalK: Int, totalN: Int) throws -> [UInt8] {
        let coeffs = try rsEngine.decode(shards: samples, originalK: originalK, totalN: totalN)
        return unpackBabyBearToBytes(Array(coeffs.prefix(originalK)))
    }

    /// Full DAS simulation: encode, sample, verify, reconstruct.
    public func simulate(data: [UInt8], sampleRatio: Double = 0.75) throws -> (
        encodedShards: Int,
        sampledShards: Int,
        verified: Bool,
        reconstructed: Bool,
        originalSize: Int,
        reconstructedSize: Int
    ) {
        let (shards, originalK, totalN) = try encodeBlob(data: data)

        let numSamples = max(originalK, Int(Double(totalN) * sampleRatio))
        let samples = sampleShards(allShards: shards, numSamples: numSamples)

        let verified = verifySamples(samples: samples, allShards: shards)

        let recovered = try reconstruct(samples: samples, originalK: originalK, totalN: totalN)

        let reconstructed = (recovered == data)

        return (
            encodedShards: totalN,
            sampledShards: samples.count,
            verified: verified,
            reconstructed: reconstructed,
            originalSize: data.count,
            reconstructedSize: recovered.count
        )
    }
}

// MARK: - Byte Packing

/// Pack raw bytes into BabyBear elements (3 bytes per element).
/// BabyBear p = 2^31 - 2^27 + 1 > 2^24, so 3 bytes (24 bits) fit safely.
func packBytesToBabyBear(_ data: [UInt8]) -> [Bb] {
    let bytesPerElement = 3
    let numElements = (data.count + bytesPerElement - 1) / bytesPerElement
    var result = [Bb](repeating: .zero, count: numElements + 1)

    // First element stores the original byte count for unambiguous unpacking
    result[0] = Bb(v: UInt32(data.count))

    for i in 0..<numElements {
        var val: UInt32 = 0
        for j in 0..<bytesPerElement {
            let byteIdx = i * bytesPerElement + j
            if byteIdx < data.count {
                val |= UInt32(data[byteIdx]) << (j * 8)
            }
        }
        result[i + 1] = Bb(v: val)
    }
    return result
}

/// Unpack BabyBear elements back to raw bytes.
func unpackBabyBearToBytes(_ elements: [Bb]) -> [UInt8] {
    guard elements.count >= 2 else { return [] }

    let originalLen = Int(elements[0].v)
    let bytesPerElement = 3
    var result = [UInt8]()
    result.reserveCapacity(originalLen)

    for i in 1..<elements.count {
        let val = elements[i].v
        for j in 0..<bytesPerElement {
            if result.count < originalLen {
                result.append(UInt8((val >> (j * 8)) & 0xFF))
            }
        }
    }
    return result
}

// MARK: - Danksharding-style DA Sampler (BN254)

/// Data availability sampler using BN254 Fr RS encoding with KZG proofs.
/// Demonstrates the Ethereum Danksharding DA sampling pattern:
/// 1. Encode blob with RS redundancy via GPU NTT
/// 2. Commit to polynomial via KZG
/// 3. Generate per-position opening proofs
/// 4. Verify individual proofs without seeing the full data
public class DankshardingSamplerBN254 {
    public let rsEngine: ReedSolomonBN254Engine
    public let redundancyFactor: Int

    public init(kzgEngine: KZGEngine? = nil, redundancyFactor: Int = 2) throws {
        self.rsEngine = try ReedSolomonBN254Engine(kzgEngine: kzgEngine)
        self.redundancyFactor = redundancyFactor
    }

    /// Encode a blob (raw bytes) into RS codeword over BN254 Fr.
    /// Packs 31 bytes per field element (BN254 Fr is ~254 bits).
    public func encodeBlob(data: [UInt8]) throws -> (codeword: [Fr], originalK: Int, totalN: Int) {
        let elements = packBytesToBN254Fr(data)
        let k = elements.count
        let codeword = try rsEngine.encode(data: elements, redundancyFactor: redundancyFactor)
        return (codeword: codeword, originalK: k, totalN: codeword.count)
    }

    /// Generate KZG commitment and per-position proofs for sampled indices.
    public func generateProofs(data: [Fr], indices: [Int]) throws -> (commitment: PointProjective, proofs: [DAKZGProof]) {
        let commitment = try rsEngine.commit(data: data)
        var proofs = [DAKZGProof]()
        proofs.reserveCapacity(indices.count)
        for idx in indices {
            let proof = try rsEngine.generateProof(data: data, index: idx)
            proofs.append(proof)
        }
        return (commitment: commitment, proofs: proofs)
    }

    /// Reconstruct original data from sampled positions.
    public func reconstruct(samples: [(index: Int, value: Fr)], originalK: Int) throws -> [UInt8] {
        let coeffs = try rsEngine.decode(received: samples, originalSize: originalK)
        return unpackBN254FrToBytes(coeffs)
    }
}

// MARK: - Danksharding-style DA Sampler (BLS12-381, Ethereum-native)

/// Data availability sampler using BLS12-381 Fr, matching Ethereum Danksharding spec.
/// BLS12-381 is the native curve for Ethereum consensus layer.
/// Supports 4096-slot blobs (logN=12) as specified in EIP-4844 / Danksharding.
public class DankshardingSampler381 {
    public let rsEngine: ReedSolomon381Engine
    public let redundancyFactor: Int

    /// Initialize with optional SRS for KZG proofs.
    /// slotCount: number of blob slots (default 4096 for Danksharding).
    public init(srs: [G1Affine381] = [], redundancyFactor: Int = 2) {
        self.rsEngine = ReedSolomon381Engine(srs: srs)
        self.redundancyFactor = redundancyFactor
    }

    /// Encode a blob into RS codeword over BLS12-381 Fr.
    /// Packs 31 bytes per field element (BLS12-381 Fr is ~255 bits).
    public func encodeBlob(data: [UInt8]) -> (codeword: [Fr381], originalK: Int, totalN: Int) {
        let elements = packBytesToBLS381Fr(data)
        let k = elements.count
        let codeword = rsEngine.encode(data: elements, redundancyFactor: redundancyFactor)
        return (codeword: codeword, originalK: k, totalN: codeword.count)
    }

    /// Generate KZG commitment and proofs on BLS12-381.
    public func generateProofs(data: [Fr381], indices: [Int]) throws -> (commitment: G1Projective381, proofs: [DAKZG381Proof]) {
        let commitment = try rsEngine.commit(data: data)
        var proofs = [DAKZG381Proof]()
        proofs.reserveCapacity(indices.count)
        for idx in indices {
            let proof = try rsEngine.generateProof(data: data, index: idx)
            proofs.append(proof)
        }
        return (commitment: commitment, proofs: proofs)
    }

    /// Reconstruct original data from sampled positions.
    public func reconstruct(samples: [(index: Int, value: Fr381)], originalK: Int) throws -> [UInt8] {
        let coeffs = try rsEngine.decode(received: samples, originalSize: originalK)
        return unpackBLS381FrToBytes(coeffs)
    }

    /// Sample random codeword indices.
    public func randomIndices(codewordSize: Int, numSamples: Int) -> [Int] {
        var indices = [Int]()
        var used = Set<Int>()
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern)
        while indices.count < min(numSamples, codewordSize) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 33) % codewordSize
            if !used.contains(idx) {
                used.insert(idx)
                indices.append(idx)
            }
        }
        return indices
    }
}

// MARK: - BN254 Fr Byte Packing

/// Pack raw bytes into BN254 Fr elements (31 bytes per element).
/// BN254 Fr is ~254 bits, so 31 bytes (248 bits) fit safely.
func packBytesToBN254Fr(_ data: [UInt8]) -> [Fr] {
    let bytesPerElement = 31
    let numElements = (data.count + bytesPerElement - 1) / bytesPerElement
    var result = [Fr](repeating: .zero, count: numElements + 1)

    // First element stores the original byte count
    result[0] = frFromInt(UInt64(data.count))

    for i in 0..<numElements {
        var limbs: [UInt64] = [0, 0, 0, 0]
        for j in 0..<bytesPerElement {
            let byteIdx = i * bytesPerElement + j
            if byteIdx < data.count {
                let limbIdx = j / 8
                let bitPos = (j % 8) * 8
                limbs[limbIdx] |= UInt64(data[byteIdx]) << bitPos
            }
        }
        let raw = Fr.from64(limbs)
        result[i + 1] = frMul(raw, Fr.from64(Fr.R2_MOD_R)) // to Montgomery
    }
    return result
}

/// Unpack BN254 Fr elements back to raw bytes.
func unpackBN254FrToBytes(_ elements: [Fr]) -> [UInt8] {
    guard elements.count >= 2 else { return [] }
    let originalLen = Int(frToUInt64(elements[0]))
    let bytesPerElement = 31
    var result = [UInt8]()
    result.reserveCapacity(originalLen)

    for i in 1..<elements.count {
        let limbs = frToInt(elements[i])
        for j in 0..<bytesPerElement {
            if result.count < originalLen {
                let limbIdx = j / 8
                let bitPos = (j % 8) * 8
                result.append(UInt8((limbs[limbIdx] >> bitPos) & 0xFF))
            }
        }
    }
    return result
}

// MARK: - BLS12-381 Fr Byte Packing

/// Pack raw bytes into BLS12-381 Fr elements (31 bytes per element).
/// BLS12-381 Fr is ~255 bits, so 31 bytes (248 bits) fit safely.
func packBytesToBLS381Fr(_ data: [UInt8]) -> [Fr381] {
    let bytesPerElement = 31
    let numElements = (data.count + bytesPerElement - 1) / bytesPerElement
    var result = [Fr381](repeating: .zero, count: numElements + 1)

    // First element stores the original byte count
    result[0] = fr381FromInt(UInt64(data.count))

    for i in 0..<numElements {
        var limbs: [UInt64] = [0, 0, 0, 0]
        for j in 0..<bytesPerElement {
            let byteIdx = i * bytesPerElement + j
            if byteIdx < data.count {
                let limbIdx = j / 8
                let bitPos = (j % 8) * 8
                limbs[limbIdx] |= UInt64(data[byteIdx]) << bitPos
            }
        }
        let raw = Fr381.from64(limbs)
        result[i + 1] = fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R)) // to Montgomery
    }
    return result
}

/// Unpack BLS12-381 Fr elements back to raw bytes.
func unpackBLS381FrToBytes(_ elements: [Fr381]) -> [UInt8] {
    guard elements.count >= 2 else { return [] }
    let limbs0 = fr381ToInt(elements[0])
    let originalLen = Int(limbs0[0])
    let bytesPerElement = 31
    var result = [UInt8]()
    result.reserveCapacity(originalLen)

    for i in 1..<elements.count {
        let limbs = fr381ToInt(elements[i])
        for j in 0..<bytesPerElement {
            if result.count < originalLen {
                let limbIdx = j / 8
                let bitPos = (j % 8) * 8
                result.append(UInt8((limbs[limbIdx] >> bitPos) & 0xFF))
            }
        }
    }
    return result
}
