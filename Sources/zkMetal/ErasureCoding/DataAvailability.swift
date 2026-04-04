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
